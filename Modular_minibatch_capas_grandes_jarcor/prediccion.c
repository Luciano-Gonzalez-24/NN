#include <stdlib.h>
#include <stdio.h>
#include "estructuras.h"
#include "prediccion.h"


float* NN_prediction_relu(NeuralNetwork* neuralnet, float* input, int print_bool){


	// NO SE PUEDE HACER ESTO SIN ANTES HACER EL ENTRENAMIENTO
	// (se podria arreglar lo de que asumimos que cosas como output ya tienen memoria reservada
	// pero meh. solo no hacer esto antes del entrenamiento)
	// (en un futuro que se puedan guardar pesos y cargarlos directamente seria interesante)


	// ARREGLAR ESTO, NO CAMBIAR LO DE LAS LAYERS,
	// Ahora tenemos el output expandido para cada ex en el training
	//
	// Si ahora usamos uno cualquiera, ¿no debería pasar nada no?
	//
	// Para seguir aprovechando que el input de las layers apunta
	// al output de la anteriro
	//
	// Usar ex = 0 por simplicidad debería funcionar
	
	int num_layers = neuralnet->num_layers;

        for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
          DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
          int input_size = Layer->input_size;
          int output_size = Layer->output_size;
	  float *input_actual = (layer_indx == 0) ? input : &neuralnet->layers[layer_indx-1].output[0];
       		
          for (int i = 0; i < output_size; i++) {
            float layer_output_aux = 0;

            for (int j = 0; j < input_size; j++) {
              layer_output_aux += Layer->weights[i * input_size + j] * input_actual[j];
            }

            layer_output_aux += Layer->bias[i];

            layer_output_aux = (layer_indx != num_layers - 1) * ((layer_output_aux>0) - layer_output_aux) + layer_output_aux;

            Layer->output[i] = layer_output_aux;

          }


        }

	if (print_bool != 0)
	{
		DenseLayer* Layer = &(neuralnet->layers[num_layers-1]);
		int output_size = Layer->output_size;
		printf("========= PREDICCION =========\n[");
		for (int i = 0; i < output_size-1 ; i++ ){
			printf("%f, ",Layer->output[i]);	
		}
		printf("%f]\n",Layer->output[output_size-1]);
	}


	DenseLayer* Layer = &(neuralnet->layers[num_layers-1]);

	return Layer->output;

}




