#include "activation.h"
#include "entrenamiento.h"
#include "estructuras.h"
#include "prediccion.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  srand48(0);
  NeuralNetwork *neuralnet = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  NN_init(neuralnet, 5);
  int input_size = 1;
  int n_examples = 1 << 20;
  int batch_size =1 << 8;
  int out_size = 1;
  int epochs = 1 << 4;
  int n_threads = 1 << 3; 
  float learning_rate = 0.01;
  float *X = (float *)malloc(input_size * n_examples * sizeof(float));
  for (int i = 0; i < n_examples; i++)
    X[i] = drand48();
  float *Y = (float *)malloc(out_size * n_examples * sizeof(float));
  for (int i = 0; i < n_examples; i++)
    Y[i] = 2 * X[i] * X[i] + 1;

  printf("Input_size :%d \nN_examples :%d \nBatch_size :%d \nout_size :%d \nEpochs :%d \nN_threads :%d \nLearning_rate :%f \n",input_size,n_examples,batch_size,out_size,epochs,n_threads,learning_rate);

  // NeuralNetwork, LayerIdx, Input_size, Output_size, Batch_size
  NN_Layer_innit(neuralnet, 0, input_size, 1 << 10,batch_size);
  NN_Layer_innit(neuralnet, 1, 1 << 10, 1 << 10,batch_size);
  NN_Layer_innit(neuralnet, 2, 1 << 10, 1 << 10,batch_size);
  NN_Layer_innit(neuralnet, 3, 1 << 10, 1 << 10,batch_size);
  NN_Layer_innit(neuralnet, 4,1 << 10,out_size,batch_size);
  double start = omp_get_wtime();
  
  // Aca le doy batch size en como argumento y no uso el de cada capa, 
  // para asi evitar llamados a cada cache de cada capa dado que son la misma siempre 
  // y porque ya lo tenia asi antes :D

  // NeuralNetwork, Epochs, X_data, Y_data, Number_Of_Examples, Batch_size, Activation_Function, Derivative_Of_Activation_Function, Learning_Rate,Number_Threads
  
  NN_loop(neuralnet, epochs, X, Y, n_examples, batch_size, Relu, Relu_prime,learning_rate,n_threads); 
  

  // recordar que dar las funciones como argumento nos quita la posibilidad de hacer static inline en las funciones
  // tal vez hacer una super funcion que ocupe un arguemneto tipo : 0, 1 ,2 para elegir la funcion a querer
  // es mas feo pero deberia ir mas rapido
  // nomas queda probarlo y ponerle static inline a lo otro, waaaaaa
  double end = omp_get_wtime();
  printf("Esto tardo %lf\n", end - start);
 // NN_print_layer(neuralnet,0);
 // NN_print_layer(neuralnet,1);
 // NN_print_layer(neuralnet,2);
 // NN_print_layer(neuralnet,3);
 //
 

  float ejemplo = 0.1;
 
 

 for (int i = 0; i<10; i++){
	NN_prediction(neuralnet,&ejemplo,1,Relu);
	ejemplo += 0.1;
 }
 

  return 0;
}
