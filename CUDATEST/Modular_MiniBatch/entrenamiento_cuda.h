#ifndef ENTRENAMIENTO_CUDA_H
#define ENTRENAMIENTO_CUDA_H

#include "estructuras.h"

#ifdef __cplusplus
extern "C" {
#endif

int NN_loop_relu_cuda(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
                      int n_examples, int batch_size, float learning_rate);

int NN_loop_sigmoid_fast_cuda(NeuralNetwork *neuralnet, int epochs, float *X,
                              float *Y, int n_examples, int batch_size,
                              float learning_rate);

#ifdef __cplusplus
}
#endif

#endif