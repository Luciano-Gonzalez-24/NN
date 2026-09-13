#ifndef ENTRENAMIENTO
#define ENTRENAMIENTO

#include "estructuras.h"

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
            int n_examples);

#endif