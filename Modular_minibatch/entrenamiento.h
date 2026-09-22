#ifndef ENTRENAMIENTO

#define ENTRENAMIENTO

#include "estructuras.h"

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples, int batch_size, float (*func)(float),
            float (*func_prime)(float));

#endif