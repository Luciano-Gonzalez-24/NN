#include "entrenamiento_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", __FILE__, __LINE__, \
              (int)status);                                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Activation types: 0 = Linear (last layer), 1 = ReLU, 2 = Fast Sigmoid
__device__ inline float activate(float z, int act_type) {
  if (act_type == 1) {
    return (z > 0.0f) ? z : 0.0f;
  } else if (act_type == 2) {
    return 0.5f * z / (1.0f + fabsf(z)) + 0.5f;
  }
  return z;
}

__device__ inline float activate_deriv(float z, int act_type) {
  if (act_type == 1) {
    return (z > 0.0f) ? 1.0f : 0.0f;
  } else if (act_type == 2) {
    float denom = 1.0f + fabsf(z);
    return 0.5f / (denom * denom);
  }
  return 1.0f;
}

// Initializes a vector of 1.0f on device (used by cublasSgemv for bias grads)
__global__ void fill_ones_kernel(float *vec, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    vec[idx] = 1.0f;
  }
}

// 1. FORWARD ELEMENT-WISE KERNEL: Adds bias to Z (from cublasSgemm) and applies activation
__global__ void add_bias_and_activate_kernel(float *__restrict__ z,
                                             const float *__restrict__ bias,
                                             float *__restrict__ a,
                                             int batch_size, int output_size,
                                             int act_type) {
  int neuron = blockIdx.x * blockDim.x + threadIdx.x;
  int ex = blockIdx.y * blockDim.y + threadIdx.y;

  if (neuron < output_size && ex < batch_size) {
    int idx = ex * output_size + neuron;
    float val = z[idx] + bias[neuron];
    z[idx] = val;
    a[idx] = activate(val, act_type);
  }
}

// 2. OUTPUT ERROR KERNEL: Computes delta for the last layer (MSE derivative)
__global__ void output_delta_kernel(const float *__restrict__ output,
                                    const float *__restrict__ target,
                                    float *__restrict__ delta_out,
                                    int batch_size, int final_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ex = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < final_size && ex < batch_size) {
    int idx = ex * final_size + i;
    delta_out[idx] = output[idx] - target[idx];
  }
}

// 3. BACKWARD ELEMENT-WISE KERNEL: Multiplies propagated delta by f'(z_prev)
__global__ void apply_act_deriv_kernel(float *__restrict__ delta_prev,
                                       const float *__restrict__ z_prev,
                                       int batch_size, int prev_size,
                                       int prev_act_type) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int ex = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < prev_size && ex < batch_size) {
    int idx = ex * prev_size + j;
    delta_prev[idx] *= activate_deriv(z_prev[idx], prev_act_type);
  }
}

// Unified internal training runner using cuBLAS
static int NN_loop_cuda_internal(NeuralNetwork *neuralnet, int epochs, float *X,
                                 float *Y, int n_examples, int batch_size,
                                 float learning_rate, int hidden_act_type) {
  int num_layers = neuralnet->num_layers;
  long long total_w = neuralnet->total_weights;
  long long total_b = neuralnet->total_biases;
  int input_dim = neuralnet->layers[0].input_size;
  int output_dim = neuralnet->layers[num_layers - 1].output_size;

  if (total_w > INT_MAX || total_b > INT_MAX) {
    fprintf(stderr, "Error: Parameter count exceeds 32-bit cuBLAS indexing.\n");
    return -1;
  }

  // Initialize cuBLAS handle
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Compute total activations across all layers for 1 batch
  long long total_batch_activations = 0;
  long long *act_offsets = (long long *)malloc(num_layers * sizeof(long long));
  for (int l = 0; l < num_layers; l++) {
    act_offsets[l] = total_batch_activations;
    total_batch_activations +=
        (long long)neuralnet->layers[l].output_size * batch_size;
  }

  // Allocate Device Memory
  float *d_X, *d_Y;
  float *d_weights, *d_bias;
  float *d_grad_w, *d_grad_b;
  float *d_z, *d_a, *d_delta;
  float *d_ones;

  CUDA_CHECK(cudaMalloc(&d_X, (long long)n_examples * input_dim * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&d_Y, (long long)n_examples * output_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, total_w * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, total_b * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_w, total_w * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_b, total_b * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, total_batch_activations * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a, total_batch_activations * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_delta, total_batch_activations * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ones, batch_size * sizeof(float)));

  int block1D = 256;
  fill_ones_kernel<<<(batch_size + block1D - 1) / block1D, block1D>>>(
      d_ones, batch_size);

  // Copy dataset to GPU once
  CUDA_CHECK(cudaMemcpy(d_X, X, (long long)n_examples * input_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Y, Y,
                        (long long)n_examples * output_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Copy initial weights and biases from host structs into flat GPU buffers
  for (int l = 0; l < num_layers; l++) {
    DenseLayer *L = &neuralnet->layers[l];
    long long w_size = (long long)L->input_size * L->output_size;
    CUDA_CHECK(cudaMemcpy(d_weights + neuralnet->weights_starter_point[l],
                          L->weights, w_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias + neuralnet->bias_starter_point[l], L->bias,
                          L->output_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Keep host output allocated so prediccion.c works after training
    L->output = (float *)calloc(L->output_size * batch_size, sizeof(float));
  }

  int num_batches = n_examples / batch_size;
  float neg_lr_over_batch = -(learning_rate / (float)batch_size);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  dim3 block2D(16, 16);

  // Training Loop
  for (int ep = 0; ep < epochs; ep++) {
    for (int batch = 0; batch < num_batches; batch++) {
      const float *batch_X = d_X + (long long)batch * batch_size * input_dim;
      const float *batch_Y = d_Y + (long long)batch * batch_size * output_dim;

      // --- FORWARD PASS ---
      for (int l = 0; l < num_layers; l++) {
        int in_sz = neuralnet->layers[l].input_size;
        int out_sz = neuralnet->layers[l].output_size;
        const float *in_ptr = (l == 0) ? batch_X : (d_a + act_offsets[l - 1]);
        const float *w_ptr = d_weights + neuralnet->weights_starter_point[l];
        const float *b_ptr = d_bias + neuralnet->bias_starter_point[l];
        float *z_ptr = d_z + act_offsets[l];
        float *a_ptr = d_a + act_offsets[l];
        int act = (l == num_layers - 1) ? 0 : hidden_act_type;

        // Z = X * W^T  (in col-major: Z_col[out_sz x B] = W_col^T[out_sz x in_sz] * X_col[in_sz x B])
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 out_sz, batch_size, in_sz,
                                 &alpha,
                                 w_ptr, in_sz,
                                 in_ptr, in_sz,
                                 &beta,
                                 z_ptr, out_sz));

        // Z += b; A = activate(Z)
        dim3 grid((out_sz + 15) / 16, (batch_size + 15) / 16);
        add_bias_and_activate_kernel<<<grid, block2D>>>(
            z_ptr, b_ptr, a_ptr, batch_size, out_sz, act);
      }

      // --- BACKWARD PASS ---
      // 1. Last layer delta (output - target)
      int last = num_layers - 1;
      dim3 grid_out((output_dim + 15) / 16, (batch_size + 15) / 16);
      output_delta_kernel<<<grid_out, block2D>>>(
          d_a + act_offsets[last], batch_Y, d_delta + act_offsets[last],
          batch_size, output_dim);

      // 2. Propagate gradients backward through all layers
      for (int l = num_layers - 1; l >= 0; l--) {
        int in_sz = neuralnet->layers[l].input_size;
        int out_sz = neuralnet->layers[l].output_size;
        const float *in_ptr = (l == 0) ? batch_X : (d_a + act_offsets[l - 1]);
        const float *w_ptr = d_weights + neuralnet->weights_starter_point[l];
        float *delta_curr = d_delta + act_offsets[l];
        float *grad_w_ptr = d_grad_w + neuralnet->weights_starter_point[l];
        float *grad_b_ptr = d_grad_b + neuralnet->bias_starter_point[l];

        // dW = delta^T * X  (in col-major: dW_col[in_sz x out_sz] = X_col[in_sz x B] * delta_col^T[B x out_sz])
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 in_sz, out_sz, batch_size,
                                 &alpha,
                                 in_ptr, in_sz,
                                 delta_curr, out_sz,
                                 &beta,
                                 grad_w_ptr, in_sz));

        // db = delta^T * 1_B  (in col-major: db[out_sz] = delta_col[out_sz x B] * ones[B])
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                                 out_sz, batch_size,
                                 &alpha,
                                 delta_curr, out_sz,
                                 d_ones, 1,
                                 &beta,
                                 grad_b_ptr, 1));

        // delta_prev = (delta_curr * W) * f'(z_prev)
        if (l > 0) {
          float *delta_prev = d_delta + act_offsets[l - 1];
          const float *z_prev = d_z + act_offsets[l - 1];

          // Linear step: delta_prev_col[in_sz x B] = W_col[in_sz x out_sz] * delta_curr_col[out_sz x B]
          CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   in_sz, batch_size, out_sz,
                                   &alpha,
                                   w_ptr, in_sz,
                                   delta_curr, out_sz,
                                   &beta,
                                   delta_prev, in_sz));

          // Element-wise activation derivative
          dim3 grid_prev((in_sz + 15) / 16, (batch_size + 15) / 16);
          apply_act_deriv_kernel<<<grid_prev, block2D>>>(
              delta_prev, z_prev, batch_size, in_sz, hidden_act_type);
        }
      }

      // --- SGD UPDATE (W -= (lr / B) * dW, b -= (lr / B) * db) ---
      CUBLAS_CHECK(cublasSaxpy(handle, (int)total_w, &neg_lr_over_batch,
                               d_grad_w, 1, d_weights, 1));
      CUBLAS_CHECK(cublasSaxpy(handle, (int)total_b, &neg_lr_over_batch,
                               d_grad_b, 1, d_bias, 1));
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy trained weights and biases back to CPU structs
  for (int l = 0; l < num_layers; l++) {
    DenseLayer *L = &neuralnet->layers[l];
    long long w_size = (long long)L->input_size * L->output_size;
    CUDA_CHECK(cudaMemcpy(L->weights,
                          d_weights + neuralnet->weights_starter_point[l],
                          w_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L->bias, d_bias + neuralnet->bias_starter_point[l],
                          L->output_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Cleanup
  CUBLAS_CHECK(cublasDestroy(handle));
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_weights);
  cudaFree(d_bias);
  cudaFree(d_grad_w);
  cudaFree(d_grad_b);
  cudaFree(d_z);
  cudaFree(d_a);
  cudaFree(d_delta);
  cudaFree(d_ones);
  free(act_offsets);

  return 0;
}

extern "C" int NN_loop_relu_cuda(NeuralNetwork *neuralnet, int epochs, float *X,
                                 float *Y, int n_examples, int batch_size,
                                 float learning_rate) {
  return NN_loop_cuda_internal(neuralnet, epochs, X, Y, n_examples, batch_size,
                               learning_rate, 1);
}

extern "C" int NN_loop_sigmoid_fast_cuda(NeuralNetwork *neuralnet, int epochs,
                                         float *X, float *Y, int n_examples,
                                         int batch_size, float learning_rate) {
  return NN_loop_cuda_internal(neuralnet, epochs, X, Y, n_examples, batch_size,
                               learning_rate, 2);
}