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
  double fw_max = 0.0;
  double bw_max = 0.0;
  double *t_fw_threads = (double *)calloc(n_threads, sizeof(double));
  double *t_bw_threads = (double *)calloc(n_threads, sizeof(double));
  double *t_update_threads = (double *)calloc(n_threads, sizeof(double));
  double *loss_threads = (double *)calloc(n_threads, sizeof(double));
  float *threads_grad_weights = calloc((long long) n_threads * neuralnet->total_weights, sizeof(float));
  float *threads_grad_bias = calloc((long long) n_threads * neuralnet->total_biases, sizeof(float));

  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc((long long)final_size * batch_size * sizeof(float));

  long long total_w = neuralnet->total_weights;
  long long total_b = neuralnet->total_biases;
  int T_used = n_threads;

  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc((long long)output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);
  float lr_over_batch = learning_rate / (float)batch_size;
  int n_batches = n_examples / batch_size; 
 
 #pragma omp parallel num_threads(n_threads)
  {
    int thread_id = omp_get_thread_num();
    int T = omp_get_num_threads();
    if (thread_id == 0)
      T_used = T;
    mkl_set_num_threads_local(1); 

    int first_row = (int)((long long)batch_size * thread_id / T);
    int last_row = (int)((long long)batch_size * (thread_id + 1) / T);
    int n_rows = last_row - first_row;

    float *grad_weights = &threads_grad_weights[(long long)thread_id * total_w];
    float *grad_bias = &threads_grad_bias[(long long)thread_id * total_b];

    double my_fw = 0.0, my_bw = 0.0, my_update = 0.0;

    for (int ep = 0; ep < epochs; ep++) {
      double loss_local = 0.0;

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
        float *input = &x[(long long)first_row * input_size];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
        for (int ex = first_row; ex < last_row; ex++) {
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
          float *input = &neuralnet->layers[layer_indx - 1].output[(long long)first_row * input_size];
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                      1.0f, input, input_size, Layer->weights, input_size,
                      0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
          for (int ex = first_row; ex < last_row; ex++) {
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
        input = &neuralnet->layers[num_layers - 2].output[(long long)first_row * input_size];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
            Layer->cache.z[ex * output_size + i] = layer_output_aux;

            layer_output_aux = layer_output_aux;

            Layer->output[ex * output_size + i] = layer_output_aux;
          }
        }

        double t_bw_start = omp_get_wtime();
        my_fw += (t_bw_start - t_fw_start);

        // BACKWARD
        grad = grad_buffer;
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < final_size; i++) {
            float diff = (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                          y[ex * final_size + i]);

            grad[ex * final_size + i] = diff;
            loss_local += 0.5 * (double)diff * diff;
          }
        }

        Layer = &(neuralnet->layers[num_layers - 1]);
        input_size = Layer->input_size;
        output_size = Layer->output_size;
        input = &neuralnet->layers[num_layers - 2].output[(long long)first_row * input_size];
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] = 0;
        for (int ex = first_row; ex < last_row; ex++)
          for (int i = 0; i < output_size; i++)
            grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] += grad[ex * output_size + i];

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                    1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[num_layers - 1]], input_size);


        DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers - 2]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows, input_size, output_size,
                    1.0f, &grad[(long long)first_row * output_size], output_size, Layer->weights, input_size,
                    0.0f, &Previous_Layer->cache.grad[(long long)first_row * input_size], input_size);
        grad = Previous_Layer->cache.grad; 
        for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
          DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
          float *input = &neuralnet->layers[layer_indx - 1].output[(long long)first_row * input_size];

          for (int i = 0; i < output_size; i++)
            grad_bias[neuralnet->bias_starter_point[layer_indx] + i] = 0;
          for (int ex = first_row; ex < last_row; ex++) {
            for (int i = 0; i < output_size; i++) {
              grad[ex * output_size + i] *= (Layer->cache.z[ex * output_size + i] > 0);
              grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad[ex * output_size + i];
            }
          }
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                      1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                      0.0f, &grad_weights[neuralnet->weights_starter_point[layer_indx]], input_size);

          DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows, input_size, output_size,
                      1.0f, &grad[(long long)first_row * output_size], output_size, Layer->weights, input_size,
                      0.0f, &Previous_Layer->cache.grad[(long long)first_row * input_size], input_size);
          grad = Previous_Layer->cache.grad; 
        } // layers

        Layer = &(neuralnet->layers[0]);
        input_size = Layer->input_size;
        output_size = Layer->output_size;
        input = &x[(long long)first_row * input_size];
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[0] + i] = 0;
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < output_size; i++) {
            grad[ex * output_size + i] *= (Layer->cache.z[ex * output_size + i] > 0);
            grad_bias[neuralnet->bias_starter_point[0] + i] += grad[ex * output_size + i];
          }
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                    1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[0]], input_size);

        double t_bw_end = omp_get_wtime();
        my_bw += (t_bw_end - t_bw_start);

        #pragma omp barrier 
        double t0 = omp_get_wtime();
        for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
          DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
          long long b_start = neuralnet->bias_starter_point[layer_indx];
          long long w_start = neuralnet->weights_starter_point[layer_indx];
          long long total_layer_weights = (long long)input_size * output_size;

          #pragma omp for nowait
          for (int i = 0; i < output_size; i++) {
            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
              sum += threads_grad_bias[(long long)t * total_b + b_start + i];
            }
            Layer->bias[i] -= lr_over_batch * sum;
          }
          long long n = total_layer_weights;
          long long k0 = n * thread_id / T;
          long long k1 = n * (thread_id + 1) / T;

          for (int t = 0; t < T; t++) {
            float *aux_grad_w = &threads_grad_weights[(long long)t * total_w + w_start];
            for (long long k = k0; k < k1; k++)
              Layer->weights[k] -= lr_over_batch * aux_grad_w[k];
          }
        }

        #pragma omp barrier 
        my_update += (omp_get_wtime() - t0);
      } // batch

      loss_threads[thread_id] = loss_local;
      #pragma omp barrier
      #pragma omp master
      {
        double loss_avg = 0.0;
        for (int t = 0; t < T; t++)
          loss_avg += loss_threads[t];
        loss_avg = loss_avg / n_examples / final_size;
        printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
      }
    } // epochs

    t_fw_threads[thread_id] = my_fw;
    t_bw_threads[thread_id] = my_bw;
    t_update_threads[thread_id] = my_update;
  } // omp

  double t_total = omp_get_wtime() - t_start_total;

  for (int t = 0; t < T_used; t++) {
    t_fw_total += t_fw_threads[t];
    t_bw_total += t_bw_threads[t];
    t_update_total += t_update_threads[t];
    if (t_fw_threads[t] > fw_max) fw_max = t_fw_threads[t];
    if (t_bw_threads[t] > bw_max) bw_max = t_bw_threads[t];
  }
  double t_fw_wall = t_fw_total / T_used;
  double t_bw_wall = t_bw_total / T_used;
  double t_update_wall = t_update_total / T_used;
  double t_rest = t_total - t_fw_wall - t_bw_wall - t_update_wall;


  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (promedio por hilo):   %.4f s (%.2f%%)  max hilo: %.4f s\n", t_fw_wall,
         (t_fw_wall / t_total) * 100.0, fw_max);
  printf("  - Backward Pass (promedio por hilo):  %.4f s (%.2f%%)  max hilo: %.4f s\n", t_bw_wall,
         (t_bw_wall / t_total) * 100.0, bw_max);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_wall, (t_update_wall / t_total) * 100.0);
  printf("  - Resto (espera en barrera 1/overhead): %.4f s (%.2f%%)\n",
         t_rest, (t_rest / t_total) * 100.0);
  printf("CSV,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n_threads, t_total,
         t_fw_wall, t_bw_wall, t_update_wall, t_rest, fw_max, bw_max);
  printf("=================================================================\n");


  free(grad_buffer);
  free(t_fw_threads);
  free(t_bw_threads);
  free(t_update_threads);
  free(loss_threads);
  free(threads_grad_weights);
  free(threads_grad_bias);

  return 0;
}


int NN_loop_sigmoid_fast(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float learning_rate, int n_threads) {

  int num_layers = neuralnet->num_layers;

  double t_start_total = omp_get_wtime();
  double t_fw_total = 0.0;
  double t_bw_total = 0.0;
  double t_update_total = 0.0;
  double fw_max = 0.0;
  double bw_max = 0.0;
  double *t_fw_threads = (double *)calloc(n_threads, sizeof(double));
  double *t_bw_threads = (double *)calloc(n_threads, sizeof(double));
  double *t_update_threads = (double *)calloc(n_threads, sizeof(double));
  double *loss_threads = (double *)calloc(n_threads, sizeof(double));
  float *threads_grad_weights = calloc((long long) n_threads * neuralnet->total_weights, sizeof(float));
  float *threads_grad_bias = calloc((long long) n_threads * neuralnet->total_biases, sizeof(float));

  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc((long long)final_size * batch_size * sizeof(float));

  long long total_w = neuralnet->total_weights;
  long long total_b = neuralnet->total_biases;
  int T_used = n_threads;



  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc((long long)output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);
  float lr_over_batch = learning_rate / (float)batch_size;
  int n_batches = n_examples / batch_size; 
  #pragma omp parallel num_threads(n_threads)
  {
    int thread_id = omp_get_thread_num();
    int T = omp_get_num_threads();
    if (thread_id == 0)
      T_used = T;
    mkl_set_num_threads_local(1); 

    int first_row = (int)((long long)batch_size * thread_id / T);
    int last_row = (int)((long long)batch_size * (thread_id + 1) / T);
    int n_rows = last_row - first_row;

    float *grad_weights = &threads_grad_weights[(long long)thread_id * total_w];
    float *grad_bias = &threads_grad_bias[(long long)thread_id * total_b];

    double my_fw = 0.0, my_bw = 0.0, my_update = 0.0;

    for (int ep = 0; ep < epochs; ep++) {
      double loss_local = 0.0;

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
        float *input = &x[(long long)first_row * input_size];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
        for (int ex = first_row; ex < last_row; ex++) {
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
          float *input = &neuralnet->layers[layer_indx - 1].output[(long long)first_row * input_size];
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                      1.0f, input, input_size, Layer->weights, input_size,
                      0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
          for (int ex = first_row; ex < last_row; ex++) {
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
        input = &neuralnet->layers[num_layers - 2].output[(long long)first_row * input_size];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, output_size, input_size,
                    1.0f, input, input_size, Layer->weights, input_size,
                    0.0f, &Layer->cache.z[(long long)first_row * output_size], output_size);
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = Layer->cache.z[ex * output_size + i] + Layer->bias[i];
            Layer->cache.z[ex * output_size + i] = layer_output_aux;

            layer_output_aux = layer_output_aux;

            Layer->output[ex * output_size + i] = layer_output_aux;
          }
        }

        double t_bw_start = omp_get_wtime();
        my_fw += (t_bw_start - t_fw_start);

        // BACKWARD
        grad = grad_buffer;
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < final_size; i++) {
            float diff = (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                          y[ex * final_size + i]);

            grad[ex * final_size + i] = diff;
            loss_local += 0.5 * (double)diff * diff;
          }
        }


        Layer = &(neuralnet->layers[num_layers - 1]);
        input_size = Layer->input_size;
        output_size = Layer->output_size;
        input = &neuralnet->layers[num_layers - 2].output[(long long)first_row * input_size];
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] = 0;
        for (int ex = first_row; ex < last_row; ex++)
          for (int i = 0; i < output_size; i++)
            grad_bias[neuralnet->bias_starter_point[num_layers - 1] + i] += grad[ex * output_size + i];
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                    1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[num_layers - 1]], input_size);


        DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers - 2]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows, input_size, output_size,
                    1.0f, &grad[(long long)first_row * output_size], output_size, Layer->weights, input_size,
                    0.0f, &Previous_Layer->cache.grad[(long long)first_row * input_size], input_size);
        grad = Previous_Layer->cache.grad; 
        for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
          DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
          float *input = &neuralnet->layers[layer_indx - 1].output[(long long)first_row * input_size];

          for (int i = 0; i < output_size; i++)
            grad_bias[neuralnet->bias_starter_point[layer_indx] + i] = 0;
          for (int ex = first_row; ex < last_row; ex++) {
            for (int i = 0; i < output_size; i++) {
              float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
    grad[ex * output_size + i] *= 0.5f / (mini_aux * mini_aux);
              grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad[ex * output_size + i];
            }
          }
          cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                      1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                      0.0f, &grad_weights[neuralnet->weights_starter_point[layer_indx]], input_size);

          DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows, input_size, output_size,
                      1.0f, &grad[(long long)first_row * output_size], output_size, Layer->weights, input_size,
                      0.0f, &Previous_Layer->cache.grad[(long long)first_row * input_size], input_size);
          grad = Previous_Layer->cache.grad; 
                                             
        } // layers

        Layer = &(neuralnet->layers[0]);
        input_size = Layer->input_size;
        output_size = Layer->output_size;
        input = &x[(long long)first_row * input_size];
        for (int i = 0; i < output_size; i++)
          grad_bias[neuralnet->bias_starter_point[0] + i] = 0;
        for (int ex = first_row; ex < last_row; ex++) {
          for (int i = 0; i < output_size; i++) {
            float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
    grad[ex * output_size + i] *= 0.5f / (mini_aux * mini_aux);
            grad_bias[neuralnet->bias_starter_point[0] + i] += grad[ex * output_size + i];
          }
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, input_size, n_rows,
                    1.0f, &grad[(long long)first_row * output_size], output_size, input, input_size,
                    0.0f, &grad_weights[neuralnet->weights_starter_point[0]], input_size);

        double t_bw_end = omp_get_wtime();
        my_bw += (t_bw_end - t_bw_start);

        #pragma omp barrier  
        double t0 = omp_get_wtime();
        for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
          DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
          long long b_start = neuralnet->bias_starter_point[layer_indx];
          long long w_start = neuralnet->weights_starter_point[layer_indx];
          long long total_layer_weights = (long long)input_size * output_size;

          #pragma omp for nowait
          for (int i = 0; i < output_size; i++) {
            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
              sum += threads_grad_bias[(long long)t * total_b + b_start + i];
            }
            Layer->bias[i] -= lr_over_batch * sum;
          }
          long long n = total_layer_weights;
          long long k0 = n * thread_id / T;
          long long k1 = n * (thread_id + 1) / T;

          for (int t = 0; t < T; t++) {
            float *aux_grad_w = &threads_grad_weights[(long long)t * total_w + w_start];
            for (long long k = k0; k < k1; k++)
              Layer->weights[k] -= lr_over_batch * aux_grad_w[k];
          }
        }

        #pragma omp barrier // 
        my_update += (omp_get_wtime() - t0);
      } // batch

      loss_threads[thread_id] = loss_local;
      #pragma omp barrier
      #pragma omp master
      {
        double loss_avg = 0.0;
        for (int t = 0; t < T; t++)
          loss_avg += loss_threads[t];
        loss_avg = loss_avg / n_examples / final_size;
        printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
      }
    } // epochs

    t_fw_threads[thread_id] = my_fw;
    t_bw_threads[thread_id] = my_bw;
    t_update_threads[thread_id] = my_update;
  } // omp

  double t_total = omp_get_wtime() - t_start_total;

  for (int t = 0; t < T_used; t++) {
    t_fw_total += t_fw_threads[t];
    t_bw_total += t_bw_threads[t];
    t_update_total += t_update_threads[t];
    if (t_fw_threads[t] > fw_max) fw_max = t_fw_threads[t];
    if (t_bw_threads[t] > bw_max) bw_max = t_bw_threads[t];
  }
  double t_fw_wall = t_fw_total / T_used;
  double t_bw_wall = t_bw_total / T_used;
  double t_update_wall = t_update_total / T_used;
  double t_rest = t_total - t_fw_wall - t_bw_wall - t_update_wall;

  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (promedio por hilo):   %.4f s (%.2f%%)  max hilo: %.4f s\n", t_fw_wall,
         (t_fw_wall / t_total) * 100.0, fw_max);
  printf("  - Backward Pass (promedio por hilo):  %.4f s (%.2f%%)  max hilo: %.4f s\n", t_bw_wall,
         (t_bw_wall / t_total) * 100.0, bw_max);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_wall, (t_update_wall / t_total) * 100.0);
  printf("  - Resto (espera en barrera 1/overhead): %.4f s (%.2f%%)\n",
         t_rest, (t_rest / t_total) * 100.0);
  printf("CSV,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n_threads, t_total,
         t_fw_wall, t_bw_wall, t_update_wall, t_rest, fw_max, bw_max);
  printf("=================================================================\n");


  free(grad_buffer);
  free(t_fw_threads);
  free(t_bw_threads);
  free(t_update_threads);
  free(loss_threads);
  free(threads_grad_weights);
  free(threads_grad_bias);

  return 0;
}
