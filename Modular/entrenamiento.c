#include "entrenamiento.h"
#include "activation.h"
#include "estructuras.h"
#include <stdio.h>
#include <stdlib.h>

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples) {
  int num_layers = neuralnet->num_layers;
  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *loss = (float *)malloc(final_size * sizeof(float));
  float *grad_buffer = (float *)malloc(final_size * sizeof(float));

  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    int input_size = neuralnet->layers[layer_indx].input_size;
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
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORWARD
    for (int ex = 0; ex < n_examples; ex++) {
      float *grad = grad_buffer;
      float *x = X + ex * neuralnet->layers[0].input_size;
      float *y = Y + ex * neuralnet->layers[num_layers - 1].output_size;
      neuralnet->layers[0].input = x;
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
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

      // BACKWARD
      // #pragma omp parallel for num_threads(16) reduction(+ : loss_avg)
      for (int i = 0; i < final_size; i++) {
        float diff = neuralnet->layers[num_layers - 1].output[i] - y[i];
        loss[i] = 0.5 * (diff) * (diff);
        grad[i] = (diff);
        loss_avg += (loss[i] / n_examples) / final_size;
        // printf("La red da %f y la respuesta es %f loss = %f\n",
        //        neuralnet->layers[num_layers - 1].output[i], y[i], loss[i]);
      }
      for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--) {
        int input_size = neuralnet->layers[layer_indx].input_size;
        int output_size = neuralnet->layers[layer_indx].output_size;

        for (int i = 0; i < output_size; i++) {
          // No olvidar que cuando tenga funcion de activacion
          // tengo que hacer grad_z*d funcion activacion
          float grad_z =
              grad[i] * Relu_prime(neuralnet->layers[layer_indx].cache.z[i]);
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
              grad_z *= Relu_prime(neuralnet->layers[layer_indx].cache.z[i]);
              // grad_z *= (neuralnet->layers[layer_indx].cache.z[i] > 0);
              // optimizacion pa dps con el el codigo mjr
              sum += grad_z *
                     neuralnet->layers[layer_indx].weights[i * input_size + j];
            }
            neuralnet->layers[layer_indx - 1].cache.grad[j] = sum;
          }
          grad = neuralnet->layers[layer_indx - 1].cache.grad;
        }
      }

      float learning_rate = 0.01;

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
    }
    printf("En la epoca %d, el loss es: %lf \n", ep, loss_avg);
    //   if (loss_avg <= 0.00002)
    //     return 0;
    // }

    return 0;
  }