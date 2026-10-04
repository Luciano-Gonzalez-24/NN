#include "estructuras.h"
#include "entrenamiento.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

float Relu(float input) { return (input > 0) * input; }

float Relu_prime(float input) { return (input > 0) * 1.0f; }

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples) {
  int num_layers = neuralnet->num_layers;
  int final_size = neuralnet->layers[num_layers - 1].output_size;

  float *loss = (float *)malloc(final_size * sizeof(float));
  float *grad_buffer = (float *)malloc(final_size * sizeof(float));

  // Acumuladores de tiempo global para el perfilado
  double t_forward_hidden = 0.0;
  double t_forward_output = 0.0;
  double t_backward_output = 0.0;
  double t_backward_hidden_and_update = 0.0;

  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    int output_size = neuralnet->layers[layer_indx].output_size;
    if (layer_indx == 0) {
      neuralnet->layers[layer_indx].output =
          (float *)calloc(output_size, sizeof(float));
    } else {
      neuralnet->layers[layer_indx].input =
          neuralnet->layers[layer_indx - 1].output;
      neuralnet->layers[layer_indx].output =
          (float *)calloc(output_size, sizeof(float));
    }
  }

  for (int ep = 0; ep < epochs; ep++) {
    float loss_avg = 0;

    for (int ex = 0; ex < n_examples; ex++) {
      float *grad = grad_buffer;
      float *x = X + ex * neuralnet->layers[0].input_size;
      float *y = Y + ex * neuralnet->layers[num_layers - 1].output_size;
      neuralnet->layers[0].input = x;

      // =========================================================================
      // 1. FORWARD PART 1: Capas Ocultas (Con ReLU)
      // =========================================================================
      double t0 = omp_get_wtime();
      for (int layer_indx = 0; layer_indx < num_layers - 1; layer_indx++) {
        int input_size = neuralnet->layers[layer_indx].input_size;
        int output_size = neuralnet->layers[layer_indx].output_size;
        for (int i = 0; i < output_size; i++) {
          neuralnet->layers[layer_indx].output[i] = 0;
          for (int j = 0; j < input_size; j++) {
            neuralnet->layers[layer_indx].output[i] +=
                neuralnet->layers[layer_indx].weights[i * input_size + j] *
                neuralnet->layers[layer_indx].input[j];
          }
          neuralnet->layers[layer_indx].output[i] +=
              neuralnet->layers[layer_indx].bias[i];
          neuralnet->layers[layer_indx].cache.z[i] =
              neuralnet->layers[layer_indx].output[i];
          neuralnet->layers[layer_indx].output[i] =
              Relu(neuralnet->layers[layer_indx].output[i]);
        }
      }
      double t1 = omp_get_wtime();
      t_forward_hidden += (t1 - t0);

      // =========================================================================
      // 2. FORWARD PART 2: Capa Final de Salida (Sin Activación)
      // =========================================================================
      int last_idx = num_layers - 1;
      int last_input_size = neuralnet->layers[last_idx].input_size;
      int last_output_size = neuralnet->layers[last_idx].output_size;

      for (int i = 0; i < last_output_size; i++) {
        neuralnet->layers[last_idx].output[i] = 0;
        for (int j = 0; j < last_input_size; j++) {
          neuralnet->layers[last_idx].output[i] +=
              neuralnet->layers[last_idx].weights[i * last_input_size + j] *
              neuralnet->layers[last_idx].input[j];
        }
        neuralnet->layers[last_idx].output[i] +=
            neuralnet->layers[last_idx].bias[i];
        neuralnet->layers[last_idx].cache.z[i] =
            neuralnet->layers[last_idx].output[i];
      }
      double t2 = omp_get_wtime();
      t_forward_output += (t2 - t1);

      // =========================================================================
      // 3. BACKWARD PART 1: Cálculo de Pérdida y Gradientes en Capa de Salida
      // =========================================================================
      for (int i = 0; i < final_size; i++) {
        loss[i] = 0.5f * (neuralnet->layers[num_layers - 1].output[i] - y[i]) *
                  (neuralnet->layers[num_layers - 1].output[i] - y[i]);
        grad[i] = (neuralnet->layers[num_layers - 1].output[i] - y[i]);
        loss_avg += (loss[i] / n_examples) / final_size;
      }
      double t3 = omp_get_wtime();
      t_backward_output += (t3 - t2);

      // =========================================================================
      // 4. BACKWARD PART 2: Retropropagación en Capas Ocultas y Actualización
      // =========================================================================
      for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--) {
        int input_size = neuralnet->layers[layer_indx].input_size;
        int output_size = neuralnet->layers[layer_indx].output_size;
        for (int i = 0; i < output_size; i++) {
          float grad_z =
              grad[i] *
              ((layer_indx != num_layers - 1) *
                   (Relu_prime(neuralnet->layers[layer_indx].cache.z[i]) - 1) +
               1);
          neuralnet->layers[layer_indx].grad_bias[i] = grad_z;
          for (int j = 0; j < input_size; j++) {
            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] =
                grad_z * neuralnet->layers[layer_indx].input[j];
          }
        }

        if (layer_indx > 0) {
          for (int j = 0; j < input_size; j++) {
            float sum = 0;
            for (int i = 0; i < output_size; i++) {
              float grad_z = grad[i];
              grad_z *=
                  (layer_indx != num_layers - 1) *
                      (Relu_prime(neuralnet->layers[layer_indx].cache.z[i]) -
                       1) +
                  1;
              sum += grad_z *
                     neuralnet->layers[layer_indx].weights[i * input_size + j];
            }
            neuralnet->layers[layer_indx - 1].cache.grad[j] = sum;
          }
          grad = neuralnet->layers[layer_indx - 1].cache.grad;
        }
      }

      // Actualización de pesos
      float learning_rate = 0.01f;
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
        int input_size = neuralnet->layers[layer_indx].input_size;
        int output_size = neuralnet->layers[layer_indx].output_size;
        for (int i = 0; i < output_size; i++) {
          neuralnet->layers[layer_indx].bias[i] -=
              learning_rate * neuralnet->layers[layer_indx].grad_bias[i];
          for (int j = 0; j < input_size; j++) {
            neuralnet->layers[layer_indx].weights[i * input_size + j] -=
                learning_rate *
                neuralnet->layers[layer_indx].grad_weights[i * input_size + j];
          }
        }
      }
      double t4 = omp_get_wtime();
      t_backward_hidden_and_update += (t4 - t3);
    }

    printf("En la epoca %d, el loss es: %lf \n", ep, loss_avg);
  }

  // Impresión detallada del perfilado de tiempos
  double t_total = t_forward_hidden + t_forward_output + t_backward_output +
                   t_backward_hidden_and_update;

  printf("\n=================== PERFILADO DE TIEMPOS ===================\n");
  printf("Forward Part 1 (Capas Ocultas)   : %.6f s (%.2f%%)\n",
         t_forward_hidden, (t_forward_hidden / t_total) * 100);
  printf("Forward Part 2 (Capa Salida)     : %.6f s (%.2f%%)\n",
         t_forward_output, (t_forward_output / t_total) * 100);
  printf("Backward Part 1 (Loss & Output)  : %.6f s (%.2f%%)\n",
         t_backward_output, (t_backward_output / t_total) * 100);
  printf("Backward Part 2 (Propag. & Pesos): %.6f s (%.2f%%)\n",
         t_backward_hidden_and_update,
         (t_backward_hidden_and_update / t_total) * 100);
  printf("Tiempo Total Computado           : %.6f s\n", t_total);
  printf("============================================================\n");

  free(loss);
  free(grad_buffer);

  return 0;
}