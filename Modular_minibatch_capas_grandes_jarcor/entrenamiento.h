#ifndef ENTRENAMIENTO

#define ENTRENAMIENTO

#include "estructuras.h"

int NN_loop_relu(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size,float learning_rate,int n_threads);

#endif