#ifndef ENTRENAMIENTO

#define ENTRENAMIENTO

float Relu(float input);

float Relu_prime(float input);

int NN_loop(NeuralNetwork* neuralnet,int epochs,float *X,float *Y,int n_examples,int batch_size);

#endif