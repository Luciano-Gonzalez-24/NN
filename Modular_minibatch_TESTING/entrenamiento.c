#include "entrenamiento.h"
#include "activation.h"
#include "estructuras.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y, int n_examples, int batch_size, float (*func)(float), float (*func_prime)(float),float learning_rate ,int n_threads) {

  int num_layers = neuralnet->num_layers;
  int final_size = neuralnet->layers[num_layers - 1].output_size;
  float *grad_buffer = (float *)malloc(final_size *batch_size *  sizeof(float));

  float *grad_weights = neuralnet->grad_weights;
  float *grad_bias = neuralnet->grad_bias;
  int total_w = neuralnet->total_weights;
  int total_b = neuralnet->total_biases;
 
  for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
    DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
    int output_size = Layer->output_size;
    Layer->output = (float *)calloc(output_size*batch_size, sizeof(float));
   
  }

  DenseLayer* First_Layer = &(neuralnet->layers[0]);
  DenseLayer* Last_Layer = &(neuralnet->layers[num_layers-1]);

  for (int ep = 0; ep < epochs; ep++) {

    float loss_avg = 0;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FORWARD

    ////////// BATCH //////////////
    for (int batch = 0; batch < (int)(n_examples / batch_size); batch++) { // Asumiendo que batch_size divide a n_examples

// Ver lo del memset, si copia bien.

      memset(grad_weights,0,neuralnet->total_weights*sizeof(float));
      memset(grad_bias,0,neuralnet->total_biases*sizeof(float));

      # pragma omp parallel for num_threads(n_threads) reduction(+:grad_weights[0:total_w],grad_bias[0:total_b],loss_avg)
      for (int ex = 0; ex < batch_size; ex++) {
        float *grad = &grad_buffer[ex*final_size];
        float *x = X + (batch * batch_size + ex) * First_Layer ->input_size;
        float *y = Y + (batch * batch_size + ex) * Last_Layer->output_size;
        

        for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
          DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
          float *input = (layer_indx == 0) ? x : &neuralnet->layers[layer_indx-1].output[ex*input_size];

          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = 0;

            for (int j = 0; j < input_size; j++) {
              layer_output_aux += Layer->weights[i * input_size + j] * input[j];
            }

            layer_output_aux += Layer->bias[i];
            Layer->cache.z[ex*output_size + i] = layer_output_aux;

            layer_output_aux = (layer_indx != num_layers - 1) * (func(layer_output_aux) - layer_output_aux) + layer_output_aux;

            Layer->output[ex*output_size + i] = layer_output_aux;

          }


        }

                
        // BACKWARD  (Recordar Reduction para loss_avg)
        for (int i = 0; i < final_size; i++) {
          float diff = (neuralnet->layers[num_layers - 1].output[ex* final_size + i] - y[i]);

          grad[i] = diff;
          loss_avg += (0.5 * diff * diff / n_examples) / final_size; 
          // printf("La red da %f y la respuesta es %f loss =
          // %f\n",neuralnet->layers[num_layers-1].output[i],y[i],loss[i]);
        }

        

        for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--) {
          DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          float *input = (layer_indx == 0) ? x : &neuralnet->layers[layer_indx-1].output[ex*input_size];
          int output_size = Layer->output_size; 
          for (int i = 0; i < output_size; i++) {
            float grad_z = grad[i] * ((layer_indx != num_layers - 1) * (func_prime(Layer->cache.z[ex*output_size  + i]) - 1) + 1);
            grad_bias[ neuralnet->bias_starter_point[layer_indx] + i] += grad_z;
            // No olvidar que cuando tenga funcion de activacion
            // tengo que hacer grad_z*d funcion activacion
            for (int j = 0; j < input_size; j++) {
              grad_weights[neuralnet->weights_starter_point[layer_indx] + i * input_size + j] += grad_z * input[j];
            }
          }

          if (layer_indx > 0) {
            DenseLayer* Previous_Layer = &(neuralnet->layers[layer_indx-1]);
            for (int j = 0; j < input_size; j++) {
              float sum = 0;
              for (int i = 0; i < output_size; i++) {
                float grad_z = grad[i];
                grad_z *= (layer_indx != num_layers - 1) * (func_prime(Layer->cache.z[ex*output_size + i]) -1) + 1;
                sum += grad_z * Layer->weights[i * input_size + j];
              }
              Previous_Layer->cache.grad[ex *input_size + j] = sum;
            }
            grad = &Previous_Layer->cache.grad[ex*input_size]; //output_size de la capa anterior es el input_size de la actual
          }


        } // termina for con layer_indx


      } // Termina el for de ejemplos


      // Se actualizan los pesos y biases
      
      for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
          DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size; 
        for (int i = 0; i < output_size; i++) {
          Layer->bias[i] -= learning_rate * grad_bias[neuralnet->bias_starter_point[layer_indx] + i] / batch_size;
          for (int j = 0; j < input_size; j++) {
            Layer->weights[i * input_size + j] -= learning_rate * grad_weights[neuralnet->weights_starter_point[layer_indx] + i * input_size + j] / batch_size;
          }
        }
      }

    } // Termina el for de los batches
    printf("En la epoca %d, el loss es: %lf \n", ep, loss_avg);
  } // Termina el for de las epochs

  return 0;
}