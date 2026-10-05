#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "estructuras.h"


int NN_init(NeuralNetwork* neuralnet,int num_layers){
	if (neuralnet == NULL){
		printf("No se inicializo la NN\n");
		return -1;
	}
	neuralnet->num_layers = num_layers;
	DenseLayer* Layer = (DenseLayer*)malloc(num_layers*sizeof(DenseLayer));
	if (Layer == NULL){
		printf("No se inicializo la NN\n");
		return -1;
	}
	for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
		Layer[layer_indx].initialized = 0;
	}
	neuralnet->layers = Layer;
	return 0;
}




int NN_Layer_innit_he(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size,int batch_size){
	if (layer_indx !=0 && input_size != neuralnet->layers[layer_indx-1].output_size){
		printf("El tamaño del output anterior no es igual a el input de esta capa \n");
		return -1;
	}
	if (layer_indx>neuralnet->num_layers-1){
		printf("El indice de layer  %d es mayor que la capacidad inicializada de la nn\n",layer_indx);
	}
	if (neuralnet->layers[layer_indx].initialized!=0){
		printf("%d ",layer_indx);
		printf("Esta capa ya esta inicializada compadre \n");
		return -1;
	}
	DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
	Layer->input_size = input_size;
	Layer->output_size = output_size;
	Layer->cache.batch_size = batch_size;
	Layer->weights = (float*)malloc(input_size*output_size*sizeof(float));
	Layer->bias    = (float*)malloc(output_size*sizeof(float));
	Layer->cache.grad = (float*)malloc(output_size*batch_size*sizeof(float));
	Layer->cache.z = (float*)malloc(output_size*batch_size*sizeof(float));


	float limit = sqrt(6.0f / input_size);
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			Layer->weights[i*input_size+j] = (float)drand48()*2*limit - limit; //random entre (-limit,limit)
		}
		Layer->bias[i] = 0.0f;
	}
	

	neuralnet->layers[layer_indx].initialized = 1;
	return 0;


}




int NN_Layer_innit_xavier(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size,int batch_size){
	if (layer_indx !=0 && input_size != neuralnet->layers[layer_indx-1].output_size){
		printf("El tamaño del output anterior no es igual a el input de esta capa \n");
		return -1;
	}
	if (layer_indx>neuralnet->num_layers-1){
		printf("El indice de layer  %d es mayor que la capacidad inicializada de la nn\n",layer_indx);
	}
	if (neuralnet->layers[layer_indx].initialized!=0){
		printf("%d ",layer_indx);
		printf("Esta capa ya esta inicializada compadre \n");
		return -1;
	}
	DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
	Layer->input_size = input_size;
	Layer->output_size = output_size;
	Layer->cache.batch_size = batch_size;
	Layer->weights = (float*)malloc(input_size*output_size*sizeof(float));
	Layer->bias    = (float*)malloc(output_size*sizeof(float));
	Layer->cache.grad = (float*)malloc(output_size*batch_size*sizeof(float));
	Layer->cache.z = (float*)malloc(output_size*batch_size*sizeof(float));


	float limit = sqrt(6.0f / (input_size + output_size));
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			Layer->weights[i*input_size+j] = (float)drand48()*2*limit - limit; //random entre (-limit,limit)
		}
		Layer->bias[i] = 0.0f;
	}
	

	neuralnet->layers[layer_indx].initialized = 1;
	return 0;


}






void  NN_print_layer(NeuralNetwork* neuralnet,int layer_indx){
	DenseLayer layer=neuralnet->layers[layer_indx];
	int output_size = layer.output_size;
	int input_size= layer.input_size;
	printf("Printing layer %d weights W:\n [\n",layer_indx);
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			printf("%f ",layer.weights[i*input_size+j]);
		}
		printf("\n");
	}
	printf("]\n");
	printf("Printing layer %d bias B:\n [\n",layer_indx);
	for (int i = 0; i < output_size; i++){
		printf("%f\n",layer.bias[i]);
	}
	printf("]\n");
}


void NN_grad_innit(NeuralNetwork* neuralnet){
	int num_layers = neuralnet->num_layers;
	neuralnet->weights_starter_point = (long long*)malloc(num_layers*sizeof(long long));
	neuralnet->bias_starter_point = (long long*)malloc(num_layers*sizeof(long long));

	long long w_starter_point = 0;
	long long b_starter_point = 0;

	for (int layer_idx = 0; layer_idx < num_layers; layer_idx++){
		long long input_size = neuralnet->layers[layer_idx].input_size;
		long long output_size = neuralnet->layers[layer_idx].output_size;
		neuralnet->weights_starter_point[layer_idx] = w_starter_point;
		neuralnet->bias_starter_point[layer_idx] = b_starter_point;
		w_starter_point += input_size * output_size;
		b_starter_point += output_size;
	}

	neuralnet->total_weights = w_starter_point;
	neuralnet->total_biases = b_starter_point;
}
