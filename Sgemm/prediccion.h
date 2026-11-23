#ifndef PREDICCION

#define PREDICCION

#include "estructuras.h"

float* NN_prediction_relu(NeuralNetwork* neuralnet, float* input,int print_bool);
float* NN_prediction_sigmoid_fast(NeuralNetwork* neuralnet, float* input,int print_bool);


#endif
