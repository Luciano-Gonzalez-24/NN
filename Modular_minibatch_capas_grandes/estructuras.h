#ifndef ESTRUCTURAS

#define ESTRUCTURAS


typedef struct {
    int batch_size;
    float *z;
	float *grad;
} DenseCache;



typedef struct {
	int input_size;
   int output_size;
// FORWARD
   float *weights; // i,j donde i es la neurona y j es el input
   float *bias; // i, hay uno por neurona
	float *output;

	int initialized;
   DenseCache cache;
} DenseLayer;



typedef struct {
   int num_layers;
   DenseLayer *layers;
   // BACKWARD
   long long *weights_starter_point;
   long long *bias_starter_point;
   long long total_weights;
   long long total_biases;
} NeuralNetwork;


int NN_init(NeuralNetwork* neuralnet,int num_layers);

int NN_Layer_innit(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size, int batch_size);

void  NN_print_layer(NeuralNetwork* neuralnet,int layer_indx);

void NN_grad_innit(NeuralNetwork* neuralnet);

#endif