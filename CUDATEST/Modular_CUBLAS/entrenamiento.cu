#include "entrenamiento_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Set to 0 to disable profiling instrumentation and summary output
#define ENABLE_PROFILING 1

#if ENABLE_PROFILING
  #if defined(__has_include)
    #if __has_include(<nvtx3/nvToolsExt.h>)
      #include <nvtx3/nvToolsExt.h>
      #define NVTX_PUSH(name) nvtxRangePushA(name)
      #define NVTX_POP() nvtxRangePop()
    #elif __has_include(<nvToolsExt.h>)
      #include <nvToolsExt.h>
      #define NVTX_PUSH(name) nvtxRangePushA(name)
      #define NVTX_POP() nvtxRangePop()
    #else
      #define NVTX_PUSH(name) do {} while (0)
      #define NVTX_POP() do {} while (0)
    #endif
  #else
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
  #endif
#else
  #define NVTX_PUSH(name) do {} while (0)
  #define NVTX_POP() do {} while (0)
#endif

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
// IDK if this makes it go slower for the if, might be betteer differente functions.
__device__ inline float activate(float z, int act_type) {
  if (act_type == 1) {
    return (z > 0.0f) ? z : 0.0f;
  } else if (act_type == 2) {
    return 0.5f * z / (1.0f + fabsf(z)) + 0.5f;
  }
  return z;
}
// IDK if this makes it go slower for the if, might be betteer differente functions.

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

// 2. OUTPUT ERROR KERNEL: Computes the difference  for the last layer (MSE derivative)
__global__ void output_difference_kernel(const float *__restrict__ output,
                                    const float *__restrict__ target,
                                    float *__restrict__ difference_out,
                                    int batch_size, int final_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ex = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < final_size && ex < batch_size) {
    int idx = ex * final_size + i;
    difference_out[idx] = output[idx] - target[idx];
  }
}

// 3. BACKWARD ELEMENT-WISE KERNEL: Multiplies propagated difference by the derivative f'(z_prev)
__global__ void apply_act_deriv_kernel(float *__restrict__ difference_prev,
                                       const float *__restrict__ z_prev,
                                       int batch_size, int prev_size,
                                       int prev_act_type) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int ex = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < prev_size && ex < batch_size) {
    int idx = ex * prev_size + j;
    difference_prev[idx] *= activate_deriv(z_prev[idx], prev_act_type);
  }
}

// corre todo viste
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

#if ENABLE_PROFILING
  clock_t t_setup_start = clock();
  NVTX_PUSH("Setup_and_H2D");
#endif

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
  float *d_z, *d_a, *d_difference;
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
  CUDA_CHECK(cudaMalloc(&d_difference, total_batch_activations * sizeof(float)));
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

#if ENABLE_PROFILING
  CUDA_CHECK(cudaDeviceSynchronize());
  NVTX_POP(); // Setup_and_H2D
  double setup_ms =
      1000.0 * (double)(clock() - t_setup_start) / CLOCKS_PER_SEC;

  cudaEvent_t ev_train_start, ev_train_stop;
  CUDA_CHECK(cudaEventCreate(&ev_train_start));
  CUDA_CHECK(cudaEventCreate(&ev_train_stop));
  NVTX_PUSH("Training_Loop");
  CUDA_CHECK(cudaEventRecord(ev_train_start));
#endif

  int num_batches = n_examples / batch_size;
  float neg_lr_over_batch = -(learning_rate / (float)batch_size);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  dim3 block2D(16, 16);

  // Training Loop
  for (int ep = 0; ep < epochs; ep++) {
    NVTX_PUSH("Epoch");
    for (int batch = 0; batch < num_batches; batch++) {
      const float *batch_X = d_X + (long long)batch * batch_size * input_dim;
      const float *batch_Y = d_Y + (long long)batch * batch_size * output_dim;

      // --- FORWARD PASS ---
      NVTX_PUSH("Forward");
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
      NVTX_POP(); // Forward

      // --- BACKWARD PASS ---
      NVTX_PUSH("Backward");
      // 1. Last layer difference (output - target)
      int last = num_layers - 1;
      dim3 grid_out((output_dim + 15) / 16, (batch_size + 15) / 16);
      output_difference_kernel<<<grid_out, block2D>>>(
          d_a + act_offsets[last], batch_Y, d_difference + act_offsets[last],
          batch_size, output_dim);

      // 2. Propagate gradients backward through all layers
      for (int l = num_layers - 1; l >= 0; l--) {
        int in_sz = neuralnet->layers[l].input_size;
        int out_sz = neuralnet->layers[l].output_size;
        const float *in_ptr = (l == 0) ? batch_X : (d_a + act_offsets[l - 1]);
        const float *w_ptr = d_weights + neuralnet->weights_starter_point[l];
        float *difference_curr = d_difference + act_offsets[l];
        float *grad_w_ptr = d_grad_w + neuralnet->weights_starter_point[l];
        float *grad_b_ptr = d_grad_b + neuralnet->bias_starter_point[l];

        // dW = difference^T * X  (in col-major: dW_col[in_sz x out_sz] = X_col[in_sz x B] * difference_col^T[B x out_sz])
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 in_sz, out_sz, batch_size,
                                 &alpha,
                                 in_ptr, in_sz,
                                 difference_curr, out_sz,
                                 &beta,
                                 grad_w_ptr, in_sz));

        // db = difference^T * 1_B  (in col-major: db[out_sz] = difference_col[out_sz x B] * ones[B])
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                                 out_sz, batch_size,
                                 &alpha,
                                 difference_curr, out_sz,
                                 d_ones, 1,
                                 &beta,
                                 grad_b_ptr, 1));

        // difference_prev = (difference_curr * W) * f'(z_prev)
        if (l > 0) {
          float *difference_prev = d_difference + act_offsets[l - 1];
          const float *z_prev = d_z + act_offsets[l - 1];

          // Linear step: difference_prev_col[in_sz x B] = W_col[in_sz x out_sz] * difference_curr_col[out_sz x B]
          CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   in_sz, batch_size, out_sz,
                                   &alpha,
                                   w_ptr, in_sz,
                                   difference_curr, out_sz,
                                   &beta,
                                   difference_prev, in_sz));

          // Element-wise activation derivative
          dim3 grid_prev((in_sz + 15) / 16, (batch_size + 15) / 16);
          apply_act_deriv_kernel<<<grid_prev, block2D>>>(
              difference_prev, z_prev, batch_size, in_sz, hidden_act_type);
        }
      }
      NVTX_POP(); // Backward

      // --- SGD UPDATE (W -= (lr / B) * dW, b -= (lr / B) * db) ---
      NVTX_PUSH("SGD_Update");
      CUBLAS_CHECK(cublasSaxpy(handle, (int)total_w, &neg_lr_over_batch,
                               d_grad_w, 1, d_weights, 1));
      CUBLAS_CHECK(cublasSaxpy(handle, (int)total_b, &neg_lr_over_batch,
                               d_grad_b, 1, d_bias, 1));
      NVTX_POP(); // SGD_Update
    }
    NVTX_POP(); // Epoch
  }

#if ENABLE_PROFILING
  CUDA_CHECK(cudaEventRecord(ev_train_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_train_stop));
  NVTX_POP(); // Training_Loop

  float train_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&train_ms, ev_train_start, ev_train_stop));
  CUDA_CHECK(cudaEventDestroy(ev_train_start));
  CUDA_CHECK(cudaEventDestroy(ev_train_stop));

  clock_t t_cleanup_start = clock();
  NVTX_PUSH("D2H_and_Cleanup");
#else
  CUDA_CHECK(cudaDeviceSynchronize());
#endif

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
  cudaFree(d_difference);
  cudaFree(d_ones);
  free(act_offsets);

#if ENABLE_PROFILING
  NVTX_POP(); // D2H_and_Cleanup
  double cleanup_ms =
      1000.0 * (double)(clock() - t_cleanup_start) / CLOCKS_PER_SEC;
  double total_samples_processed =
      (double)epochs * (double)(num_batches * batch_size);
  double throughput = (train_ms > 0.0f)
                          ? (total_samples_processed / (train_ms / 1000.0))
                          : 0.0;

  printf("\n=== CUDA Training Profile Summary ===\n");
  printf("Setup & H2D Transfer : %10.3f ms\n", setup_ms);
  printf("Pure GPU Training    : %10.3f ms (%d epochs, %d batches/epoch)\n",
         train_ms, epochs, num_batches);
  printf("  -> Time per Epoch  : %10.3f ms\n",
         (epochs > 0) ? (train_ms / epochs) : 0.0);
  printf("  -> Time per Batch  : %10.4f ms\n",
         (epochs * num_batches > 0) ? (train_ms / (epochs * num_batches)) : 0.0);
  printf("  -> Throughput      : %10.0f samples/sec\n", throughput);
  printf("D2H Copy & Cleanup   : %10.3f ms\n", cleanup_ms);
  printf("Total End-to-End     : %10.3f ms\n",
         setup_ms + train_ms + cleanup_ms);
  printf("=====================================\n\n");
#endif

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