#include "entrenamiento.h"
#include "estructuras.h"
#include <mkl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int NN_loop_relu(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float learning_rate, int n_threads) {

  int num_layers = neuralnet->num_layers;
  
  double t_start_total = omp_get_wtime();
  double t_fw_total = 0.0;
  double t_bw_total = 0.0;
  double t_update_total = 0.0;

  float *grad_weights = calloc((long long) neuralnet->total_weights, sizeof(float));
  float *grad_bias = calloc((long long) neuralnet->total_biases, sizeof(float));

  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc((long long)final_size * batch_size * sizeof(float));

  
    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc((long long)output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);
  float lr_over_batch = learning_rate / (float)batch_size;
  int n_batches = n_examples / batch_size; 

  mkl_set_num_threads(n_threads);

  for (int ep = 0; ep < epochs; ep++) {
    double loss_avg = 0;

    ////////// BATCH //////////////
    for (int batch = 0; batch < n_batches; batch++) {
      double t_fw_start = omp_get_wtime();

      float *x = &X[(long long)batch * batch_size * First_Layer->input_size];
      float *y = &Y[(long long)batch * batch_size * Last_Layer->output_size];
      float *grad;

      // FORWARD


      DenseLayer *Layer = &(neuralnet->layers[0]);
      int input_size = Layer->input_size;
      int output_size = Layer->output_size;
      float *input = x;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                  1.0f, input, input_size, Layer->weights, input_size,
                  0.0f, Layer->cache.z, output_size);
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
          Layer->cache.z[ex * output_size + i] = layer_output_aux;

          layer_output_aux = (layer_output_aux > 0) * layer_output_aux;

          Layer->output[ex * output_size + i] = layer_output_aux;
        }
      }

      for (int layer_indx = 1; layer_indx < num_layers - 1; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float *input = neuralnet->layers[layer_indx - 1].output;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, Layer->cache.z, output_size);
        for (int ex = 0; ex < batch_size; ex++) {
          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
            Layer->cache.z[ex * output_size + i] = layer_output_aux;

            layer_output_aux = (layer_output_aux > 0) * layer_output_aux;

            Layer->output[ex * output_size + i] = layer_output_aux;
          }
        }
      }

      Layer = &(neuralnet->layers[num_layers - 1]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = neuralnet->layers[num_layers - 2].output;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                  1.0f, input, input_size, Layer->weights, input_size,
                  0.0f, Layer->cache.z, output_size);
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
          Layer->cache.z[ex * output_size + i] = layer_output_aux;

          layer_output_aux = layer_output_aux;

          Layer->output[ex * output_size + i] = layer_output_aux;
        }
      }

      double t_bw_start = omp_get_wtime();
      t_fw_total += (t_bw_start - t_fw_start);

      // BACKWARD
      grad = grad_buffer;
      double loss_batch = 0.0;
      #pragma omp parallel for num_threads(n_threads) reduction(+: loss_batch) if ((long long)batch_size * final_size >= NN_PAR_MIN)
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < final_size; i++) {
          float diff = (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                        y[ex * final_size + i]);

          grad[ex * final_size + i] = diff;
          loss_batch += 0.5 * (double)diff * diff;
        }
      }
      loss_avg += loss_batch;

      Layer = &(neuralnet->layers[num_layers - 1]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = neuralnet->layers[num_layers - 2].output;
      for (int i = 0; i < output_size; i++)
        grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] = 0;
      for (int ex = 0; ex < batch_size; ex++)
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] += grad[ex * output_size + i];
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                  1.0f, grad, output_size, input, input_size,
                  0.0f, &grad_weights[neuralnet->weights_starter_point[num_layers - 1]], input_size);

      DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers - 2]);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, input_size, output_size,
                  1.0f, grad, output_size, Layer->weights, input_size,
                  0.0f, Previous_Layer->cache.grad, input_size);
      grad = Previous_Layer->cache.grad; 

      for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float *input = neuralnet->layers[layer_indx - 1].output;

        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[layer_indx] + i] = 0;
        for (int ex = 0; ex < batch_size; ex++) {
          for (int i = 0; i < output_size; i++) {
            grad[ex * output_size + i] *= (Layer->cache.z[ex * output_size + i] > 0);
            grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad[ex * output_size + i];
          }
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                    1.0f, grad, output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[layer_indx]], input_size);

        DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, input_size, output_size,
                    1.0f, grad, output_size, Layer->weights, input_size,
                    0.0f, Previous_Layer->cache.grad, input_size);
        grad = Previous_Layer->cache.grad; 
      } // layers


      Layer = &(neuralnet->layers[0]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = x;
      for (int i = 0; i < output_size; i++)
        grad_bias[neuralnet->bias_starter_point[0] + i] = 0;
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          grad[ex * output_size + i] *= (Layer->cache.z[ex * output_size + i] > 0);
          grad_bias[neuralnet->bias_starter_point[0] + i] += grad[ex * output_size + i];
        }
      }
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                  1.0f, grad, output_size, input, input_size,
                  0.0f, &grad_weights[neuralnet->weights_starter_point[0]], input_size);

      double t_bw_end = omp_get_wtime();
      t_bw_total += (t_bw_end - t_bw_start);

      double t0 = omp_get_wtime();
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        long long b_start = neuralnet->bias_starter_point[layer_indx];
        long long w_start = neuralnet->weights_starter_point[layer_indx];
        long long total_layer_weights = (long long)input_size * output_size;

        cblas_saxpy((int)total_layer_weights, -lr_over_batch, &grad_weights[w_start], 1, Layer->weights, 1);
        cblas_saxpy(output_size, -lr_over_batch, &grad_bias[b_start], 1, Layer->bias, 1);
      }
      t_update_total += (omp_get_wtime() - t0);

    } // batch

    loss_avg = loss_avg / n_examples / final_size;
    printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
  } // epochs

  double t_total = omp_get_wtime() - t_start_total;
  double t_rest = t_total - t_fw_total - t_bw_total - t_update_total;


  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (tiempo real):         %.4f s (%.2f%%)\n", t_fw_total,
         (t_fw_total / t_total) * 100.0);
  printf("  - Backward Pass (tiempo real):        %.4f s (%.2f%%)\n", t_bw_total,
         (t_bw_total / t_total) * 100.0);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_total, (t_update_total / t_total) * 100.0);
  printf("  - Resto:                              %.4f s (%.2f%%)\n",
         t_rest, (t_rest / t_total) * 100.0);
  printf("CSV,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n_threads, t_total,
         t_fw_total, t_bw_total, t_update_total, t_rest, t_fw_total, t_bw_total);
  printf("=================================================================\n");

  free(grad_buffer);
  free(grad_weights);
  free(grad_bias);

  return 0;
}


int NN_loop_sigmoid_fast(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float learning_rate, int n_threads) {

  int num_layers = neuralnet->num_layers;

  double t_start_total = omp_get_wtime();
  double t_fw_total = 0.0;
  double t_bw_total = 0.0;
  double t_update_total = 0.0;

  float *grad_weights = calloc((long long) neuralnet->total_weights, sizeof(float));
  float *grad_bias = calloc((long long) neuralnet->total_biases, sizeof(float));

  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc((long long)final_size * batch_size * sizeof(float));

  
    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc((long long)output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);
  float lr_over_batch = learning_rate / (float)batch_size;
  int n_batches = n_examples / batch_size; 


  mkl_set_num_threads(n_threads);

  for (int ep = 0; ep < epochs; ep++) {
    double loss_avg = 0;

    ////////// BATCH //////////////
    for (int batch = 0; batch < n_batches; batch++) {
      double t_fw_start = omp_get_wtime();

      float *x = &X[(long long)batch * batch_size * First_Layer->input_size];
      float *y = &Y[(long long)batch * batch_size * Last_Layer->output_size];
      float *grad;

      // FORWARD

      DenseLayer *Layer = &(neuralnet->layers[0]);
      int input_size = Layer->input_size;
      int output_size = Layer->output_size;
      float *input = x;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                  1.0f, input, input_size, Layer->weights, input_size,
                  0.0f, Layer->cache.z, output_size);
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
          Layer->cache.z[ex * output_size + i] = layer_output_aux;

          layer_output_aux = 0.5f * layer_output_aux / (1.0f + fabsf(layer_output_aux)) + 0.5f;

          Layer->output[ex * output_size + i] = layer_output_aux;
        }
      }

      for (int layer_indx = 1; layer_indx < num_layers - 1; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float *input = neuralnet->layers[layer_indx - 1].output;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, Layer->cache.z, output_size);
        for (int ex = 0; ex < batch_size; ex++) {
          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
            Layer->cache.z[ex * output_size + i] = layer_output_aux;

            layer_output_aux = 0.5f * layer_output_aux / (1.0f + fabsf(layer_output_aux)) + 0.5f;

            Layer->output[ex * output_size + i] = layer_output_aux;
          }
        }
      }

      Layer = &(neuralnet->layers[num_layers - 1]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = neuralnet->layers[num_layers - 2].output;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, input_size,
                  1.0f, input, input_size, Layer->weights, input_size,
                  0.0f, Layer->cache.z, output_size);
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
          Layer->cache.z[ex * output_size + i] = layer_output_aux;

          layer_output_aux = layer_output_aux;

          Layer->output[ex * output_size + i] = layer_output_aux;
        }
      }

      double t_bw_start = omp_get_wtime();
      t_fw_total += (t_bw_start - t_fw_start);

      // BACKWARD
      grad = grad_buffer;
      double loss_batch = 0.0;
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < final_size; i++) {
          float diff = (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                        y[ex * final_size + i]);

          grad[ex * final_size + i] = diff;
          loss_batch += 0.5 * (double)diff * diff;
        }
      }
      loss_avg += loss_batch;

      Layer = &(neuralnet->layers[num_layers - 1]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = neuralnet->layers[num_layers - 2].output;
      for (int i = 0; i < output_size; i++)
        grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] = 0;
      for (int ex = 0; ex < batch_size; ex++)
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] += grad[ex * output_size + i];
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                  1.0f, grad, output_size, input, input_size,
                  0.0f, &grad_weights[neuralnet->weights_starter_point[num_layers - 1]], input_size);

      DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers - 2]);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, input_size, output_size,
                  1.0f, grad, output_size, Layer->weights, input_size,
                  0.0f, Previous_Layer->cache.grad, input_size);
      grad = Previous_Layer->cache.grad; 

      for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float *input = neuralnet->layers[layer_indx - 1].output;

        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[layer_indx] + i] = 0;
        for (int ex = 0; ex < batch_size; ex++) {
          for (int i = 0; i < output_size; i++) {
            float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
    grad[ex * output_size + i] *= 0.5f / (mini_aux * mini_aux);
            grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad[ex * output_size + i];
          }
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                    1.0f, grad, output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[layer_indx]], input_size);

        DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, input_size, output_size,
                    1.0f, grad, output_size, Layer->weights, input_size,
                    0.0f, Previous_Layer->cache.grad, input_size);
        grad = Previous_Layer->cache.grad; 
      } // layers

      Layer = &(neuralnet->layers[0]);
      input_size = Layer->input_size;
      output_size = Layer->output_size;
      input = x;
      for (int i = 0; i < output_size; i++)
        grad_bias[neuralnet->bias_starter_point[0] + i] = 0;
      for (int ex = 0; ex < batch_size; ex++) {
        for (int i = 0; i < output_size; i++) {
          float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
    grad[ex * output_size + i] *= 0.5f / (mini_aux * mini_aux);
          grad_bias[neuralnet->bias_starter_point[0] + i] += grad[ex * output_size + i];
        }
      }
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, batch_size,
                  1.0f, grad, output_size, input, input_size,
                  0.0f, &grad_weights[neuralnet->weights_starter_point[0]], input_size);

      double t_bw_end = omp_get_wtime();
      t_bw_total += (t_bw_end - t_bw_start);

      double t0 = omp_get_wtime();
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        long long b_start = neuralnet->bias_starter_point[layer_indx];
        long long w_start = neuralnet->weights_starter_point[layer_indx];
        long long total_layer_weights = (long long)input_size * output_size;

        cblas_saxpy((int)total_layer_weights, -lr_over_batch, &grad_weights[w_start], 1, Layer->weights, 1);
        cblas_saxpy(output_size, -lr_over_batch, &grad_bias[b_start], 1, Layer->bias, 1);
      }
      t_update_total += (omp_get_wtime() - t0);

    } // batch

    loss_avg = loss_avg / n_examples / final_size;

    printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
  } // epochs

  double t_total = omp_get_wtime() - t_start_total;
  double t_rest = t_total - t_fw_total - t_bw_total - t_update_total;


  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (tiempo real):         %.4f s (%.2f%%)\n", t_fw_total,
         (t_fw_total / t_total) * 100.0);
  printf("  - Backward Pass (tiempo real):        %.4f s (%.2f%%)\n", t_bw_total,
         (t_bw_total / t_total) * 100.0);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_total, (t_update_total / t_total) * 100.0);
  printf("  - Resto:                              %.4f s (%.2f%%)\n",
         t_rest, (t_rest / t_total) * 100.0);
  printf("CSV,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n_threads, t_total,
         t_fw_total, t_bw_total, t_update_total, t_rest, t_fw_total, t_bw_total);
  printf("=================================================================\n");

  free(grad_buffer);
  free(grad_weights);
  free(grad_bias);

  return 0;
}
