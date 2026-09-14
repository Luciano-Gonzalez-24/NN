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
	float *input; //Si no soy la primera neurona, esto se puede asignar a la anterior
	float *output;
// BACKWARD
   float *grad_weights; // i,j donde i son los input y j es la neurona   (Esto es correcto  ???, no coincide con el otro comentario)
   float *grad_bias; // j, hay uno por neurona
	int initialized;
   DenseCache cache;
} DenseLayer;


typedef struct {
   int num_layers;
   DenseLayer *layers;
} NeuralNetwork;


int NN_init(NeuralNetwork* neuralnet,int num_layers);

int NN_Layer_innit(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size);

void  NN_print_layer(NeuralNetwork* neuralnet,int layer_indx);

#endif