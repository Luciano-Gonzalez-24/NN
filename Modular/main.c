#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "estructuras.h"


int main(){
	NeuralNetwork* neuralnet = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	NN_init(neuralnet,4);
	int input_size = 1;
	int n_examples = 100;
	int out_size = 1;
	float *X = (float*)malloc(input_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) X[i]=drand48();
	float *Y = (float*)malloc(out_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) Y[i]=2*X[i]*X[i]+1;
	NN_Layer_innit(neuralnet,0,input_size,8);
	NN_Layer_innit(neuralnet,1,8,8);
	NN_Layer_innit(neuralnet,2,8,8);
	NN_Layer_innit(neuralnet,3,8,out_size);
	//NN_print_layer(neuralnet,0);
	//NN_print_layer(neuralnet,1);
	double start = omp_get_wtime();
	NN_loop(neuralnet,100,X,Y,n_examples);
	double end = omp_get_wtime();
	printf("Esto tardo %lf\n",end-start);
	return 0;
}