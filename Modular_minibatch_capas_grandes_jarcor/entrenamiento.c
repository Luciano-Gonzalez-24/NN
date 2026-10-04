#include "entrenamiento.h"
#include "estructuras.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int NN_loop_relu(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float learning_rate, int n_threads) {

  double t_start_total = omp_get_wtime();
  double t_fw_total = 0.0;
  double t_bw_total = 0.0;
  double t_update_total = 0.0;
  //double* t_fw_threads = (double*)calloc(n_threads , sizeof(double));
  //double* t_bw_threads = (double*)calloc(n_threads , sizeof(double));
  float* threads_grad_weights = calloc((long long) n_threads * neuralnet->total_weights,sizeof(float));
  float* threads_grad_bias = calloc((long long) n_threads * neuralnet->total_biases,sizeof(float));


  int num_layers = neuralnet->num_layers;
  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc(final_size * batch_size * sizeof(float));

  int total_w = neuralnet->total_weights;
  int total_b = neuralnet->total_biases;

  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc(output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);

  for (int ep = 0; ep < epochs; ep++) {
    //memset(t_fw_threads,0,n_threads*sizeof(double));
    //memset(t_bw_threads,0,n_threads*sizeof(double));
    float loss_avg = 0;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORWARD

    ////////// BATCH //////////////
    for (int batch = 0; batch < (int)(n_examples / batch_size);
         batch++) { // Asumiendo que batch_size divide a n_examples

      // Ver lo del memset, si copia bien.

      double t0 = omp_get_wtime();


      #pragma omp parallel num_threads(n_threads) reduction(+: loss_avg,t_fw_total,t_bw_total)
      {
        int thread_id = omp_get_thread_num();
        memset(&threads_grad_weights[thread_id*total_w],0,total_w*sizeof(float));
        memset(&threads_grad_bias[thread_id*total_b],0,total_b*sizeof(float));
        float *grad_weights = &threads_grad_weights[thread_id*total_w];
        float *grad_bias = &threads_grad_bias[thread_id*total_b];

        #pragma omp for
        for (int ex = 0; ex < batch_size; ex++) {
          double t_fw_start = omp_get_wtime();

          float *grad = &grad_buffer[ex * final_size];
          float *x = X + (batch * batch_size + ex) * First_Layer->input_size;
          float *y = Y + (batch * batch_size + ex) * Last_Layer->output_size;







            DenseLayer *Layer = &(neuralnet->layers[0]);
            int input_size = Layer->input_size;
            int output_size = Layer->output_size;
            float *input = x;
            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

              layer_output_aux = (layer_output_aux >0)*layer_output_aux;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }








          for (int layer_indx = 1; layer_indx < num_layers-1; layer_indx++) {
            DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
            int input_size = Layer->input_size;
            int output_size = Layer->output_size;
            float *input = &neuralnet->layers[layer_indx - 1].output[ex * input_size];

            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

        layer_output_aux = (layer_output_aux > 0) * layer_output_aux;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }
          }


      Layer = &(neuralnet->layers[num_layers-1]);
            input_size = Layer->input_size;
            output_size = Layer->output_size;
            input = &neuralnet->layers[num_layers - 2].output[ex * input_size];

            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

              layer_output_aux = layer_output_aux;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }










          double t_bw_start = omp_get_wtime();
    //t_fw_threads[omp_get_thread_num()] +=(t_bw_start - t_fw_start); 
          t_fw_total += (t_bw_start - t_fw_start);

          // BACKWARD  (Recordar Reduction para loss_avg)
          for (int i = 0; i < final_size; i++) {
            float diff =
                (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                y[i]);

            grad[i] = diff;
            loss_avg += (0.5 * diff * diff / n_examples) / final_size;
            //printf("La red da %f y la respuesta es %f loss = %f\n",neuralnet->layers[num_layers-1].output[i],y[i],loss_avg);
          }















            Layer = &(neuralnet->layers[num_layers-1]);
            input_size = Layer->input_size;
            input = &neuralnet->layers[num_layers - 2].output[ex * input_size];
            output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
              float grad_z = grad[i];
              grad_bias[neuralnet->bias_starter_point[num_layers-1] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[num_layers-1] + i * input_size + j] += grad_z * input[j];
              }
            }


              DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers-2]);
              for (int j = 0; j < input_size; j++) {
                float sum = 0;
                for (int i = 0; i < output_size; i++) {
                  float grad_z = grad[i];
                  sum += grad_z * Layer->weights[i * input_size + j];
                }
                Previous_Layer->cache.grad[ex * input_size + j] = sum;
              }
              grad =&Previous_Layer->cache.grad[ex * input_size]; // output_size de la capa anterior
                                              // es el input_size de la actual














          for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
      DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
            int input_size = Layer->input_size;
            float *input = &neuralnet->layers[layer_indx - 1].output[ex * input_size];
            int output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
              float grad_z = grad[i] * (Layer->cache.z[ex * output_size + i]>0);
              grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[layer_indx] + i * input_size + j] += grad_z * input[j];
              }
            }

              DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
              for (int j = 0; j < input_size; j++) {
                float sum = 0;
                for (int i = 0; i < output_size; i++) {
                  float grad_z = grad[i];
                  grad_z *= (Layer->cache.z[ex * output_size + i]>0);
                  sum += grad_z * Layer->weights[i * input_size + j];
                }
                Previous_Layer->cache.grad[ex * input_size + j] = sum;
              }
              grad =
                  &Previous_Layer->cache
                      .grad[ex * input_size]; // output_size de la capa anterior
                                              // es el input_size de la actual
          } // termina for con layer_indx








      Layer = &(neuralnet->layers[0]);
            input_size = Layer->input_size;
            input = x;
            output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
              float grad_z = grad[i] * (Layer->cache.z[ex * output_size + i]>0);
              grad_bias[neuralnet->bias_starter_point[0] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[0] + i * input_size + j] += grad_z * input[j];
              }
            }






















          double t_bw_end = omp_get_wtime();
    //t_bw_threads[omp_get_thread_num()] +=(t_bw_end - t_bw_start);
          t_bw_total += (t_bw_end - t_bw_start);

        } // Termina el for de ejemplos
      }

      // Se actualizan los pesos y biases

      t0 = omp_get_wtime();
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float lr_over_batch = learning_rate / (float)batch_size;
        long long b_start = neuralnet->bias_starter_point[layer_indx];
        long long w_start = neuralnet->weights_starter_point[layer_indx];
        long long total_layer_weights = input_size* output_size;


        

        #pragma omp parallel num_threads(n_threads)
        {
          #pragma omp for nowait
          for(int i=0; i<output_size;i++){
            float sum = 0.0f;
            for (int t = 0; t < n_threads; t++) {
              sum += threads_grad_bias[(long long )t*total_b +b_start +i];
            }
            Layer->bias[i] -= lr_over_batch * sum;
          }
          int id = omp_get_thread_num();
          long long n  = total_layer_weights;
          long long k0 = n * id / n_threads; 
          long long k1 = n * (id + 1) / n_threads;

          for (int t = 0; t < n_threads; t++) {
            float *aux_grad_w = &threads_grad_weights[(long long)t*total_w + w_start];
            for (long long k = k0; k < k1; k++)
              Layer->weights[k] -= lr_over_batch * aux_grad_w[k];
          }
        }

        /*
        // Aplanar el for, para que no pregunte en cada iteracion
        for (int k = 0; k < total_layer_weights; k++) {
          Layer->weights[k] -= lr_over_batch * grad_weights[w_start + k];
        }
        */

      }
      t_update_total += (omp_get_wtime() - t0);

    } // Termina el for de los batches
/*
    printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
    printf("==== TIEMPO FORWARD POR HILLO EPOCH %d  ====\n[%f, ",ep,t_fw_threads[0]);
    for(int i = 1; i<n_threads-1; i++){
	printf("%f, ",t_fw_threads[i]);
    }
    printf("%f]\n",t_fw_threads[n_threads-1]);



    
    printf("==== TIEMPO BACKWARD POR HILLO EPOCH %d ====\n[%f, ",ep,t_bw_threads[0]);
    for(int i = 1; i<n_threads-1; i++){
	printf("%f, ",t_bw_threads[i]);
    }
    printf("%f]\n",t_bw_threads[n_threads-1]);
*/
  } // Termina el for de las epochs

  double t_total = omp_get_wtime() - t_start_total;

  // Promedio estimado de tiempo paralelo (se divide el tiempo acumulado entre
  // hilos)
  double t_fw_wall = t_fw_total / n_threads;
  double t_bw_wall = t_bw_total / n_threads;

  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (estimado paralelo):   %.4f s (%.2f%%)\n", t_fw_wall,
         (t_fw_wall / t_total) * 100.0);
  printf("  - Backward Pass (estimado paralelo):  %.4f s (%.2f%%)\n", t_bw_wall,
         (t_bw_wall / t_total) * 100.0);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_total, (t_update_total / t_total) * 100.0);
  printf("=================================================================\n");

  free(grad_buffer);
  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    free(neuralnet->layers[layer_indx].output);
  }
  //free(t_fw_threads);
  //free(t_bw_threads);
  free(threads_grad_weights);
  free(threads_grad_bias);

  return 0;
}
































































int NN_loop_sigmoid_fast(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float learning_rate, int n_threads) {

  double t_start_total = omp_get_wtime();
  double t_fw_total = 0.0;
  double t_bw_total = 0.0;
  double t_update_total = 0.0;
  //double* t_fw_threads = (double*)calloc(n_threads , sizeof(double));
  //double* t_bw_threads = (double*)calloc(n_threads , sizeof(double));
  float* threads_grad_weights = calloc((long long) n_threads * neuralnet->total_weights,sizeof(float));
  float* threads_grad_bias = calloc((long long) n_threads * neuralnet->total_biases,sizeof(float));


  int num_layers = neuralnet->num_layers;
  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc(final_size * batch_size * sizeof(float));

  int total_w = neuralnet->total_weights;
  int total_b = neuralnet->total_biases;

  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc(output_size * batch_size, sizeof(float));
  }

  DenseLayer *First_Layer = &(neuralnet->layers[0]);
  DenseLayer *Last_Layer = &(neuralnet->layers[num_layers - 1]);

  for (int ep = 0; ep < epochs; ep++) {
    //memset(t_fw_threads,0,n_threads*sizeof(double));
    //memset(t_bw_threads,0,n_threads*sizeof(double));
    float loss_avg = 0;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORWARD

    ////////// BATCH //////////////
    for (int batch = 0; batch < (int)(n_examples / batch_size);
         batch++) { // Asumiendo que batch_size divide a n_examples

      // Ver lo del memset, si copia bien.

      double t0 = omp_get_wtime();


      #pragma omp parallel num_threads(n_threads) reduction(+: loss_avg,t_fw_total,t_bw_total)
      {
        int thread_id = omp_get_thread_num();
        memset(&threads_grad_weights[thread_id*total_w],0,total_w*sizeof(float));
        memset(&threads_grad_bias[thread_id*total_b],0,total_b*sizeof(float));
        float *grad_weights = &threads_grad_weights[thread_id*total_w];
        float *grad_bias = &threads_grad_bias[thread_id*total_b];

        #pragma omp for
        for (int ex = 0; ex < batch_size; ex++) {
          double t_fw_start = omp_get_wtime();

          float *grad = &grad_buffer[ex * final_size];
          float *x = X + (batch * batch_size + ex) * First_Layer->input_size;
          float *y = Y + (batch * batch_size + ex) * Last_Layer->output_size;







            DenseLayer *Layer = &(neuralnet->layers[0]);
            int input_size = Layer->input_size;
            int output_size = Layer->output_size;
            float *input = x;
            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

              layer_output_aux = 0.5f * layer_output_aux / (1.0f + fabsf(layer_output_aux)) + 0.5f;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }








          for (int layer_indx = 1; layer_indx < num_layers-1; layer_indx++) {
            DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
            int input_size = Layer->input_size;
            int output_size = Layer->output_size;
            float *input = &neuralnet->layers[layer_indx - 1].output[ex * input_size];

            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

        layer_output_aux = 0.5f * layer_output_aux / (1.0f + fabsf(layer_output_aux)) + 0.5f;
;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }
          }


      Layer = &(neuralnet->layers[num_layers-1]);
            input_size = Layer->input_size;
            output_size = Layer->output_size;
            input = &neuralnet->layers[num_layers - 2].output[ex * input_size];

            for (int i = 0; i < output_size; i++) {
              float layer_output_aux = 0;

              for (int j = 0; j < input_size; j++) {
                layer_output_aux += Layer->weights[i * input_size + j] * input[j];
              }

              layer_output_aux += Layer->bias[i];
              Layer->cache.z[ex * output_size + i] = layer_output_aux;

              layer_output_aux = layer_output_aux;

              Layer->output[ex * output_size + i] = layer_output_aux;
            }










          double t_bw_start = omp_get_wtime();
    //t_fw_threads[omp_get_thread_num()] +=(t_bw_start - t_fw_start); 
          t_fw_total += (t_bw_start - t_fw_start);

          // BACKWARD  (Recordar Reduction para loss_avg)
          for (int i = 0; i < final_size; i++) {
            float diff =
                (neuralnet->layers[num_layers - 1].output[ex * final_size + i] -
                y[i]);

            grad[i] = diff;
            loss_avg += (0.5 * diff * diff / n_examples) / final_size;
            //printf("La red da %f y la respuesta es %f loss = %f\n",neuralnet->layers[num_layers-1].output[i],y[i],loss_avg);
          }















            Layer = &(neuralnet->layers[num_layers-1]);
            input_size = Layer->input_size;
            input = &neuralnet->layers[num_layers - 2].output[ex * input_size];
            output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
              float grad_z = grad[i];
              grad_bias[neuralnet->bias_starter_point[num_layers-1] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[num_layers-1] + i * input_size + j] += grad_z * input[j];
              }
            }


              DenseLayer *Previous_Layer = &(neuralnet->layers[num_layers-2]);
              for (int j = 0; j < input_size; j++) {
                float sum = 0;
                for (int i = 0; i < output_size; i++) {
                  float grad_z = grad[i];
                  sum += grad_z * Layer->weights[i * input_size + j];
                }
                Previous_Layer->cache.grad[ex * input_size + j] = sum;
              }
              grad =&Previous_Layer->cache.grad[ex * input_size]; // output_size de la capa anterior
                                              // es el input_size de la actual














          for (int layer_indx = num_layers - 2; layer_indx > 0; layer_indx--) {
      DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
            int input_size = Layer->input_size;
            float *input = &neuralnet->layers[layer_indx - 1].output[ex * input_size];
            int output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
	      float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
              float grad_z = grad[i]* 0.5f /(mini_aux * mini_aux);
              grad_bias[neuralnet->bias_starter_point[layer_indx] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[layer_indx] + i * input_size + j] += grad_z * input[j];
              }
            }

		// REvisar ESTOO

              DenseLayer *Previous_Layer = &(neuralnet->layers[layer_indx - 1]);
              for (int j = 0; j < input_size; j++) {
                float sum = 0;
                for (int i = 0; i < output_size; i++) {
	      	float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);

                  float grad_z = grad[i];
                  grad_z *= 0.5f / (mini_aux * mini_aux);
                  sum += grad_z * Layer->weights[i * input_size + j];
                }
                Previous_Layer->cache.grad[ex * input_size + j] = sum;
              }
              grad =
                  &Previous_Layer->cache
                      .grad[ex * input_size]; // output_size de la capa anterior
                                              // es el input_size de la actual
          } // termina for con layer_indx








      Layer = &(neuralnet->layers[0]);
            input_size = Layer->input_size;
            input = x;
            output_size = Layer->output_size;
            for (int i = 0; i < output_size; i++) {
	      float mini_aux = 1.0f + fabsf(Layer->cache.z[ex * output_size + i]);
              float grad_z = grad[i]* 0.5f /(mini_aux * mini_aux);
              grad_bias[neuralnet->bias_starter_point[0] + i] += grad_z;
              // No olvidar que cuando tenga funcion de activacion
              // tengo que hacer grad_z*d funcion activacion
              for (int j = 0; j < input_size; j++) {
                grad_weights[neuralnet->weights_starter_point[0] + i * input_size + j] += grad_z * input[j];
              }
            }






















          double t_bw_end = omp_get_wtime();
    //t_bw_threads[omp_get_thread_num()] +=(t_bw_end - t_bw_start);
          t_bw_total += (t_bw_end - t_bw_start);

        } // Termina el for de ejemplos
      }

      // Se actualizan los pesos y biases

      t0 = omp_get_wtime();
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
        DenseLayer *Layer = &(neuralnet->layers[layer_indx]);
        int input_size = Layer->input_size;
        int output_size = Layer->output_size;
        float lr_over_batch = learning_rate / (float)batch_size;
        long long b_start = neuralnet->bias_starter_point[layer_indx];
        long long w_start = neuralnet->weights_starter_point[layer_indx];
        long long total_layer_weights = input_size* output_size;


        

        #pragma omp parallel num_threads(n_threads)
        {
          #pragma omp for nowait
          for(int i=0; i<output_size;i++){
            float sum = 0.0f;
            for (int t = 0; t < n_threads; t++) {
              sum += threads_grad_bias[(long long )t*total_b +b_start +i];
            }
            Layer->bias[i] -= lr_over_batch * sum;
          }
          int id = omp_get_thread_num();
          long long n  = total_layer_weights;
          long long k0 = n * id / n_threads; 
          long long k1 = n * (id + 1) / n_threads;

          for (int t = 0; t < n_threads; t++) {
            float *aux_grad_w = &threads_grad_weights[(long long)t*total_w + w_start];
            for (long long k = k0; k < k1; k++)
              Layer->weights[k] -= lr_over_batch * aux_grad_w[k];
          }
        }

        /*
        // Aplanar el for, para que no pregunte en cada iteracion
        for (int k = 0; k < total_layer_weights; k++) {
          Layer->weights[k] -= lr_over_batch * grad_weights[w_start + k];
        }
        */

      }
      t_update_total += (omp_get_wtime() - t0);

    } // Termina el for de los batches
/*
    printf("\nEn la epoca %d, el loss es: %lf \n", ep, loss_avg);
    printf("==== TIEMPO FORWARD POR HILLO EPOCH %d  ====\n[%f, ",ep,t_fw_threads[0]);
    for(int i = 1; i<n_threads-1; i++){
	printf("%f, ",t_fw_threads[i]);
    }
    printf("%f]\n",t_fw_threads[n_threads-1]);



    
    printf("==== TIEMPO BACKWARD POR HILLO EPOCH %d ====\n[%f, ",ep,t_bw_threads[0]);
    for(int i = 1; i<n_threads-1; i++){
	printf("%f, ",t_bw_threads[i]);
    }
    printf("%f]\n",t_bw_threads[n_threads-1]);
*/
  } // Termina el for de las epochs

  double t_total = omp_get_wtime() - t_start_total;

  // Promedio estimado de tiempo paralelo (se divide el tiempo acumulado entre
  // hilos)
  double t_fw_wall = t_fw_total / n_threads;
  double t_bw_wall = t_bw_total / n_threads;

  printf(
      "\n================ METRICAS DE TIEMPO (PROFILING) ================\n");
  printf("Tiempo total de ejecucion:          %.4f s\n", t_total);
  printf("  - Forward Pass (estimado paralelo):   %.4f s (%.2f%%)\n", t_fw_wall,
         (t_fw_wall / t_total) * 100.0);
  printf("  - Backward Pass (estimado paralelo):  %.4f s (%.2f%%)\n", t_bw_wall,
         (t_bw_wall / t_total) * 100.0);
  printf("  - Actualizacion de pesos (SGD):       %.4f s (%.2f%%)\n",
         t_update_total, (t_update_total / t_total) * 100.0);
  printf("=================================================================\n");

  free(grad_buffer);
  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    free(neuralnet->layers[layer_indx].output);
  }
  //free(t_fw_threads);
  //free(t_bw_threads);
  free(threads_grad_weights);
  free(threads_grad_bias);

  return 0;
}
