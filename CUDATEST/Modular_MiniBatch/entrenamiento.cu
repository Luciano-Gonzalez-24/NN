#include "entrenamiento_cuda.h"
#include <cuda_runtime.h>
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

// 1. FORWARD KERNEL: Computes Z and A for one layer across the entire batch
// Grid: (ceil(output_size / 16), ceil(batch_size / 16))
__global__ void forward_layer_kernel(const float *__restrict__ input,
                                     const float *__restrict__ weights,
                                     const float *__restrict__ bias,
                                     float *__restrict__ z_out,
                                     float *__restrict__ a_out, int batch_size,
                                     int input_size, int output_size,
                                     int act_type) {
  int neuron = blockIdx.x * blockDim.x + threadIdx.x; // i
  int ex = blockIdx.y * blockDim.y + threadIdx.y;     // example in batch

  if (neuron < output_size && ex < batch_size) {
    float sum = bias[neuron];
    const float *in_row = input + ex * input_size;
    const float *w_row = weights + neuron * input_size;

    for (int j = 0; j < input_size; j++) {
      sum += w_row[j] * in_row[j];
    }

    int idx = ex * output_size + neuron;
    z_out[idx] = sum;
    a_out[idx] = activate(sum, act_type);
  }
}

// 2. OUTPUT ERROR KERNEL: Computes delta for the last layer (MSE derivative)
// Grid: (ceil(final_size / 16), ceil(batch_size / 16))
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

// 3. HIDDEN DELTA KERNEL: Backpropagates delta from layer L to layer L-1
// Grid: (ceil(prev_size / 16), ceil(batch_size / 16))
__global__ void backward_delta_kernel(const float *__restrict__ delta_curr,
                                      const float *__restrict__ weights_curr,
                                      const float *__restrict__ z_prev,
                                      float *__restrict__ delta_prev,
                                      int batch_size, int curr_size,
                                      int prev_size, int prev_act_type) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // prev neuron
  int ex = blockIdx.y * blockDim.y + threadIdx.y; // example in batch

  if (j < prev_size && ex < batch_size) {
    float sum = 0.0f;
    const float *d_row = delta_curr + ex * curr_size;

    for (int i = 0; i < curr_size; i++) {
      sum += d_row[i] * weights_curr[i * prev_size + j];
    }

    int idx = ex * prev_size + j;
    delta_prev[idx] = sum * activate_deriv(z_prev[idx], prev_act_type);
  }
}

// 4. ACCUMULATE WEIGHT GRADIENTS: Sums (delta_i * input_j) across the batch
// Grid: (ceil(input_size / 16), ceil(output_size / 16))
__global__ void grad_weights_kernel(const float *__restrict__ delta,
                                    const float *__restrict__ input,
                                    float *__restrict__ grad_w, int batch_size,
                                    int input_size, int output_size) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // input neuron
  int i = blockIdx.y * blockDim.y + threadIdx.y; // output neuron

  if (i < output_size && j < input_size) {
    float sum = 0.0f;
    for (int ex = 0; ex < batch_size; ex++) {
      sum += delta[ex * output_size + i] * input[ex * input_size + j];
    }
    grad_w[i * input_size + j] = sum;
  }
}

// 5. ACCUMULATE BIAS GRADIENTS: Sums delta_i across the batch
// Grid: (ceil(output_size / 256))
__global__ void grad_bias_kernel(const float *__restrict__ delta,
                                 float *__restrict__ grad_b, int batch_size,
                                 int output_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) {
    float sum = 0.0f;
    for (int ex = 0; ex < batch_size; ex++) {
      sum += delta[ex * output_size + i];
    }
    grad_b[i] = sum;
  }
}

// 6. SGD UPDATE KERNEL
// Grid: (ceil(n / 256))
__global__ void sgd_update_kernel(float *__restrict__ params,
                                  const float *__restrict__ grads,
                                  float lr_over_batch, long long n) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    params[idx] -= lr_over_batch * grads[idx];
  }
}

// Unified internal training runner
static int NN_loop_cuda_internal(NeuralNetwork *neuralnet, int epochs, float *X,
                                 float *Y, int n_examples, int batch_size,
                                 float learning_rate, int hidden_act_type) {
  int num_layers = neuralnet->num_layers;
  long long total_w = neuralnet->total_weights;
  long long total_b = neuralnet->total_biases;
  int input_dim = neuralnet->layers[0].input_size;
  int output_dim = neuralnet->layers[num_layers - 1].output_size;

#if ENABLE_PROFILING
  clock_t t_setup_start = clock();
  NVTX_PUSH("Setup_and_H2D");
#endif

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
  float lr_over_batch = learning_rate / (float)batch_size;
  dim3 block2D(16, 16);
  int block1D = 256;

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
        int act = (l == num_layers - 1) ? 0 : hidden_act_type;

        dim3 grid((out_sz + 15) / 16, (batch_size + 15) / 16);
        forward_layer_kernel<<<grid, block2D>>>(
            in_ptr, d_weights + neuralnet->weights_starter_point[l],
            d_bias + neuralnet->bias_starter_point[l], d_z + act_offsets[l],
            d_a + act_offsets[l], batch_size, in_sz, out_sz, act);
      }
      NVTX_POP(); // Forward

      // --- BACKWARD PASS ---
      NVTX_PUSH("Backward");
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
        float *delta_curr = d_delta + act_offsets[l];

        // Accumulate dW and db for layer l
        dim3 grid_w((in_sz + 15) / 16, (out_sz + 15) / 16);
        grad_weights_kernel<<<grid_w, block2D>>>(
            delta_curr, in_ptr, d_grad_w + neuralnet->weights_starter_point[l],
            batch_size, in_sz, out_sz);

        int grid_b = (out_sz + block1D - 1) / block1D;
        grad_bias_kernel<<<grid_b, block1D>>>(
            delta_curr, d_grad_b + neuralnet->bias_starter_point[l], batch_size,
            out_sz);

        // Compute delta for layer l - 1
        if (l > 0) {
          dim3 grid_prev((in_sz + 15) / 16, (batch_size + 15) / 16);
          backward_delta_kernel<<<grid_prev, block2D>>>(
              delta_curr, d_weights + neuralnet->weights_starter_point[l],
              d_z + act_offsets[l - 1], d_delta + act_offsets[l - 1],
              batch_size, out_sz, in_sz, hidden_act_type);
        }
      }
      NVTX_POP(); // Backward

      // --- SGD UPDATE ---
      NVTX_PUSH("SGD_Update");
      int grid_upd_w = (total_w + block1D - 1) / block1D;
      sgd_update_kernel<<<grid_upd_w, block1D>>>(d_weights, d_grad_w,
                                                 lr_over_batch, total_w);

      int grid_upd_b = (total_b + block1D - 1) / block1D;
      sgd_update_kernel<<<grid_upd_b, block1D>>>(d_bias, d_grad_b,
                                                 lr_over_batch, total_b);
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

  // Free GPU memory
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_weights);
  cudaFree(d_bias);
  cudaFree(d_grad_w);
  cudaFree(d_grad_b);
  cudaFree(d_z);
  cudaFree(d_a);
  cudaFree(d_delta);
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