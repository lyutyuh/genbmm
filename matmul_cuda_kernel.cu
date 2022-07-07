#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>
namespace {

    // FORWARD KERNELS

template <typename scalar_t>
__global__ void matmul_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < b_size) {
      scalar_t val = 0.0;

      scalar_t m = -1e9;
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
         }
      }
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         val += exp(v - m);
      }
      out[n][row][col] = log(val) + m;
      maxes[n][row][col] = m;
  }
}

template <typename scalar_t>
__global__ void max_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  int ind = -1;
  if (row < a_size && col < b_size) {
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
             ind = i;
         }
      }
      out[n][row][col] = m;
      indices[n][row][col] = ind;
  }
}

template <typename scalar_t>
__global__ void sample_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> rand,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  int ind = -1;
  if (row < a_size && col < b_size) {

      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
         }
      }
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         val += exp(v - m);
      }
      out[n][row][col] = log(val) + m;

      scalar_t total = 0.0;
      auto r = rand[n][row][col];
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col] - out[n][row][col];
         if (total < r && total + exp(v) > r ){
             indices[n][row][col] = i;
             break;
         }
         total += exp(v);
      }

  }
}


template <typename scalar_t>
__global__ void prod_max_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  int ind = -1;
  if (row < a_size && col < b_size) {
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] * b[n][i][col];
         if (v > m) {
             m = v;
             ind = i;
         }
      }
      out[n][row][col] = m;
      indices[n][row][col] = ind;
  }
}


// BACKWARD KERNELS

// LOGSUM

template <typename scalar_t>
__global__ void matmul_cuda_backward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < b_size; ++k) {
          scalar_t m = maxes[n][row][k];
          val += (exp(a[n][row][col] + b[n][col][k] - m) / (exp(part[n][row][k] -m))) * grad_output[n][row][k];
      }
      grad_a[n][row][col] = val;
  }
}


template <typename scalar_t>
__global__ void inside_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int diag
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size)
  {
      if((col - row) >= diag)
      {
          grad_a[n][row][col] = 0;
      }
      else if ((col - row) >= 0)
      {
          // valid area
          scalar_t val = 0.0;
          
          if (row + diag < a_size)
          { // horizontal use
              int k = row + diag;
              val += exp(a[n][row][col] + a[n][col+1][k] - part[n][row][k]) * grad_output[n][row][k];
          }
          if (col - diag >= 0)
          { // vertical use
              int k = col - diag;
              val += exp(a[n][k][row-1] + a[n][row][col] - part[n][k][col]) * grad_output[n][k][col];
          }
          
          grad_a[n][row][col] = val + grad_output[n][row][col];
          
      }
      else
      {
          grad_a[n][row][col] = 0;
      }
  }
  
}

template <typename scalar_t>
__global__ void inside_rule_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_rule,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> rule,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int diag
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size)
  {
      if((col - row) >= diag)
      {
          grad_a[n][row][col] = 0;
          
          if((col - row) == diag)
          {
              grad_rule[n][row][col] = 1.0 * grad_output[n][row][col];
          }
          else
          {
              grad_rule[n][row][col] = 0;
          }
      }
      else if ((col - row) >= 0)
      {
          // valid area
          scalar_t val = 0.0;
          
          if (row + diag < a_size)
          { // horizontal use
              int k = row + diag;
              val += exp(a[n][row][col] + a[n][col+1][k] - part[n][row][k] + rule[n][row][k]) * grad_output[n][row][k];
          }
          if (col - diag >= 0)
          { // vertical use
              int k = col - diag;
              val += exp(a[n][k][row-1] + a[n][row][col] - part[n][k][col] + rule[n][k][col]) * grad_output[n][k][col];
          }
          
          grad_a[n][row][col] = val + grad_output[n][row][col];
          grad_rule[n][row][col] = 0;
      }
      else
      {
          grad_a[n][row][col] = 0;
          grad_rule[n][row][col] = 0;
      }
  }
}





template <typename scalar_t>
__global__ void matmul_cuda_backward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
      scalar_t val = 0.0;
      scalar_t total = 0.0;
      for (int k = 0; k < a_size; ++k) {
          scalar_t m = maxes[n][k][col];
          scalar_t v = exp(a[n][k][row] + b[n][row][col] - m);
          val += (v / (exp(part[n][k][col] -m))) * grad_output[n][k][col];
      }
      grad_b[n][row][col] = val;
  }
}


template <typename scalar_t>
__global__ void matmul_cuda_backbackward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output_a,

    const int in_size,
    const int a_size,
    const int b_size
                                                  ) {
    const int n = blockIdx.z;
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < a_size && col < in_size) {
        scalar_t val = 0.0;
        for (int k = 0; k < b_size; ++k) {
            scalar_t m = maxes[n][row][k];
            scalar_t z = exp(part[n][row][k] -m);
            scalar_t s = exp(a[n][row][col] + b[n][col][k] - m) / z;
            scalar_t inner = 0.0;
            for (int k2 = 0; k2 < in_size; ++k2) {
                scalar_t s2 = exp(a[n][row][k2] + b[n][k2][k] - m) / z;
                scalar_t v;
                if (col == k2) {
                    v = s  - s * s2;
                } else {
                    v = - s * s2;
                }
                inner += v * grad_output_a[n][row][k2];
            }
            val += inner * grad_output[n][row][k];
        }
        grad_a[n][row][col] = val;
    }
}


template <typename scalar_t>
__global__ void matmul_cuda_backbackward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output_a,

    const int in_size,
    const int a_size,
    const int b_size
                                                  ) {


    const int n = blockIdx.z;

    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    /* n row col k, k2=>  b, t,  r, l, s*/
    if (row < in_size && col < b_size) {
        scalar_t val = 0.0;
        for (int k = 0; k < a_size; ++k) {
            scalar_t m = maxes[n][k][col];
            scalar_t z = exp(part[n][k][col] -m);
            scalar_t s = exp(a[n][k][row] + b[n][row][col] - m) / z;
            scalar_t inner = 0.0;
            for (int k2 = 0; k2 < in_size; ++k2) {
                scalar_t s2 = exp(a[n][k][k2] + b[n][k2][col] - m) / z;
                scalar_t v;
                if (row == k2) {
                    v = s  - s * s2;
                } else {
                    v = - s * s2;
                }
                inner += v * grad_output_a[n][k][k2];
            }
            val += inner * grad_output[n][k][col];
        }
        grad_b[n][row][col] = val;
    }
}

template <typename scalar_t>
__global__ void matmul_cuda_backbackward_kernel_C(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < b_size) {
      scalar_t val = 0.0;
      scalar_t m = maxes[n][row][col];
      for (int k = 0; k < in_size; ++k) {

          val += (exp(a[n][row][k] + b[n][k][col] - m) / (exp(part[n][row][col] -m))) * grad_output[n][row][k];
      }
      grad_a[n][row][col] = val;
  }
}


// MAX / SAMPLE

template <typename scalar_t>
__global__ void max_cuda_backward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < b_size; ++k) {
          /* scalar_t v = (col == part[n][row][k]) ? b[n][col][k] : 0.0; */
          /* val += v * grad_output[n][row][k]; */
          if (col == part[n][row][k]) {
              val += grad_output[n][row][k];
          }
      }
      grad_a[n][row][col] = val;
  }
}




template <typename scalar_t>
__global__ void max_cuda_backward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < a_size; ++k) {
          /* scalar_t v = (row == part[n][k][col]) ? a[n][k][row] : 0.0; */
          /* val += v * grad_output[n][k][col]; */
          if (row == part[n][k][col]) {
              val += grad_output[n][k][col];
          }
      }
      grad_b[n][row][col] = val;
  }
}


// PROD-MAX
template <typename scalar_t>
__global__ void prod_max_cuda_backward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < b_size; ++k) {
          /* scalar_t v = (col == part[n][row][k]) ? b[n][col][k] : 0.0; */
          /* val += v * grad_output[n][row][k]; */
          if (col == part[n][row][k]) {
              val += b[n][col][k] * grad_output[n][row][k];
          }
      }
      grad_a[n][row][col] = val;
  }
}


template <typename scalar_t>
__global__ void prod_max_cuda_backward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < a_size; ++k) {
          /* scalar_t v = (row == part[n][k][col]) ? a[n][k][row] : 0.0; */
          /* val += v * grad_output[n][k][col]; */
          if (row == part[n][k][col]) {
              val += a[n][k][row] * grad_output[n][k][col];
          }
      }
      grad_b[n][row][col] = val;
  }
}


template <typename scalar_t>
__global__ void inside_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    const int in_size,
    const int a_size,
    const int diag
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < a_size)
  {
      if((col - row) == diag)
      {
          scalar_t val = 0.0;
          scalar_t m = 0.0;

          for (int i = row; i < col; ++i) {
             scalar_t v = a[n][row][i] + a[n][i+1][col];
             if (v > m) {
                 val *= exp(m - v);
                 m = v;
                 val += 1;
             }
             else 
             {
                 val += exp(v - m);
             }
          }
          out[n][row][col] = log(val) + m;
      }
      else if((col - row) > diag)
      {
          out[n][row][col] = -10000.0;
      }
      else
      {
          out[n][row][col] = a[n][row][col];
      }
      
  }
}

template <typename scalar_t>
__global__ void inside_rule_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> rule,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    const int in_size,
    const int a_size,
    const int diag
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < a_size)
  {
      if((col - row) == diag)
      {
          scalar_t val = 0.0;
          scalar_t m = 0.0;

          for (int i = row; i < col; ++i) {
             scalar_t v = a[n][row][i] + a[n][i+1][col];
             if (v > m) {
                 val *= exp(m - v);
                 m = v;
                 val += 1;
             }
             else 
             {
                 val += exp(v - m);
             }
          }
          out[n][row][col] = log(val) + m + rule[n][row][col];
      }
      else if((col - row) > diag)
      {
          out[n][row][col] = -10000.0;
      }
      else
      {
          out[n][row][col] = a[n][row][col];
      }
      
  }
}

} // namespace


// MATMUL FORWARD DISPATCH
std::vector<torch::Tensor> matmul_cuda_forward(
    torch::Tensor a,
    torch::Tensor b,
    int mode) {

  const int batch_size = a.size(0);
  const int a_size = a.size(1);
  const int b_size = b.size(2);

  auto options = torch::TensorOptions()
          .dtype(a.dtype())
          .device(torch::kCUDA, a.device().index());
  auto out = torch::zeros({batch_size, a_size, b_size}, options);

  const int in_size = a.size(2);
  const int threads = 32;
  const dim3 threads_per_block(threads, threads, 1);
  const dim3 blocks(a_size / threads + 1,
                    b_size / threads + 1,
                    batch_size);

  // Dispatch
  if (mode == 0) {
      auto maxes = torch::zeros({batch_size, a_size, b_size}, options);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  matmul_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, maxes};
  } else if (mode == 1) {
      auto options2 = torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCUDA, a.device().index());
      auto indices = torch::zeros({batch_size, a_size, b_size}, options2);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, indices};
  } else if (mode == 2) {
      auto options2 = torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCUDA, a.device().index());
      auto indices = torch::zeros({batch_size, a_size, b_size}, options2);
      auto rand = torch::rand({batch_size, a_size, b_size}, options);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  sample_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      rand.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, indices};
  } else if (mode == 3) {
      auto options2 = torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCUDA, a.device().index());
      auto indices = torch::zeros({batch_size, a_size, b_size}, options2);
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  prod_max_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, indices};
  }
}

std::vector<torch::Tensor> inside_cuda_forward(
    torch::Tensor a,
    int diag) {

  const int batch_size = a.size(0);
  const int a_size = a.size(1);

  auto options = torch::TensorOptions()
          .dtype(a.dtype())
          .device(torch::kCUDA, a.device().index());
  auto out = torch::zeros({batch_size, a_size, a_size}, options);

  const int in_size = a.size(2);
  const int threads = 32;
  const dim3 threads_per_block(threads, threads, 1);
  const dim3 blocks(a_size / threads + 1,
                    a_size / threads + 1,
                    batch_size);

  // Dispatch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "inside_forward_cuda", ([&] {
              inside_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                  a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  in_size, a_size, diag);
          } ) );
  return {out};
}


std::vector<torch::Tensor> inside_cuda_backward(
    torch::Tensor a,
    torch::Tensor grad_out,
    torch::Tensor part,
    int diag) {

 const auto batch_size = a.size(0);
  const auto in_size = a.size(2);
  const int a_size = a.size(1);

  const int threads = 32;
  const dim3 blocks(a_size / threads + 1,
                    in_size / threads + 1,
                    batch_size);
  const dim3 threads_per_block(threads, threads, 1);
  auto grad_a = torch::zeros_like(a);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "inside_backward_cuda", ([&] {
              inside_cuda_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                  grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  in_size, a_size, diag);}));


  return {grad_a};
}


std::vector<torch::Tensor> inside_rule_cuda_forward(
    torch::Tensor a,
    torch::Tensor rule,
    int diag) {

  const int batch_size = a.size(0);
  const int a_size = a.size(1);

  auto options = torch::TensorOptions()
          .dtype(a.dtype())
          .device(torch::kCUDA, a.device().index());
  auto out = torch::zeros({batch_size, a_size, a_size}, options);

  const int in_size = a.size(2);
  const int threads = 32;
  const dim3 threads_per_block(threads, threads, 1);
  const dim3 blocks(a_size / threads + 1,
                    a_size / threads + 1,
                    batch_size);

  // Dispatch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "inside_rule_forward_cuda", ([&] {
              inside_rule_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                  a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  rule.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  in_size, a_size, diag);
          } ) );
  return {out};
}


std::vector<torch::Tensor> inside_rule_cuda_backward(
    torch::Tensor a,
    torch::Tensor rule,
    torch::Tensor grad_out,
    torch::Tensor part,
    int diag) {

  const auto batch_size = a.size(0);
  const auto in_size = a.size(2);
  const int a_size = a.size(1);

  const int threads = 32;
  const dim3 blocks(a_size / threads + 1,
                    in_size / threads + 1,
                    batch_size);
  const dim3 threads_per_block(threads, threads, 1);
  auto grad_a = torch::zeros_like(a);
  auto grad_rule = torch::zeros_like(rule);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "inside_rule_backward_cuda", ([&] {
              inside_rule_cuda_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                  grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  grad_rule.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  rule.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  in_size, a_size, diag);}));


  return {grad_a, grad_rule};
}



// MATMUL BACKBACKWARD DISPATCH
std::vector<torch::Tensor> matmul_cuda_backbackward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_out,
    torch::Tensor part,
    torch::Tensor maxes,
    torch::Tensor grad_out_a,
    int mode) {

  const auto batch_size = a.size(0);
  const auto in_size = a.size(2);
  const int a_size = a.size(1);
  const int b_size = b.size(2);

  const int threads = 32;
  const dim3 blocks(a_size / threads + 1,
                    in_size / threads + 1,
                    batch_size);
  const dim3 threads_per_block(threads, threads, 1);
  auto grad_a = torch::zeros_like(a);


  auto grad_b = torch::zeros_like(b);
  /* auto grad_bp = grad_b.packed_accessor32<float,3,torch::RestrictPtrTraits>(); */
  const int threads2 = 32;
  const dim3 blocks2(in_size / threads2 + 1,
                    b_size / threads2 + 1,
                    batch_size);


  auto grad_grad = torch::zeros_like(grad_out);
  const int threads3 = 32;
  const dim3 blocks3(a_size / threads3 + 1,
                     b_size / threads3 + 1,
                     batch_size);

  if (mode == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_backbackward_cuda", ([&] {
                  matmul_cuda_backbackward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size
                                                                                         );
              }));
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_backbackward_cuda", ([&] {
                  matmul_cuda_backbackward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>(
                      grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size
                                                                                         );
              }));
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_backbackward_cuda", ([&] {
                  matmul_cuda_backbackward_kernel_C<scalar_t><<<blocks3, threads_per_block>>>(
                      grad_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size
                                                                                         );
              }));

  }
  return {grad_a, grad_b, grad_grad};
}

// MATMUL BACKWARD DISPATCH
std::vector<torch::Tensor> matmul_cuda_backward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_out,
    torch::Tensor part,
    torch::Tensor maxes,
    int mode) {

  const auto batch_size = a.size(0);
  const auto in_size = a.size(2);
  const int a_size = a.size(1);
  const int b_size = b.size(2);

  const int threads = 32;
  const dim3 blocks(a_size / threads + 1,
                    in_size / threads + 1,
                    batch_size);
  const dim3 threads_per_block(threads, threads, 1);
  auto grad_a = torch::zeros_like(a);


  auto grad_b = torch::zeros_like(b);
  /* auto grad_bp = grad_b.packed_accessor32<float,3,torch::RestrictPtrTraits>(); */
  const int threads2 = 32;
  const dim3 blocks2(in_size / threads2 + 1,
                    b_size / threads2 + 1,
                    batch_size);

  if (mode == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  matmul_cuda_backward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size
                                                                                         );
              }));

      /* AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] { */
      /*             matmul_cuda_backward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>( */
      /*                 grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), */
      /*                 in_size, a_size, b_size); */
      /*         })); */
  } else if (mode == 1 or mode == 2) {

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_backward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_backward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>(
                      grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));
  } else if (mode == 3) {

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  prod_max_cuda_backward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "matmul_forward_cuda", ([&] {
                  prod_max_cuda_backward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>(
                      grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));
  }
  return {grad_a, grad_b};
}
