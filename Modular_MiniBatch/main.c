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
  int batch_size = 1 << 8;
  int out_size = 1;
  int epochs = 1 << 4;
  int n_threads = 16;
  float learning_rate = 0.01;
  float *X = (float *)malloc(input_size * n_examples * sizeof(float));
  for (int i = 0; i < n_examples; i++)
    X[i] = drand48();
  float *Y = (float *)malloc(out_size * n_examples * sizeof(float));
  for (int i = 0; i < n_examples; i++)
    Y[i] = 2 * X[i] * X[i] + 1;

  printf("Input_size :%d \nN_examples :%d \nBatch_size :%d \nout_size :%d "
         "\nEpochs :%d \nN_threads :%d \nLearning_rate :%f \n",
         input_size, n_examples, batch_size, out_size, epochs, n_threads,
         learning_rate);

  int function_select, weights_select;

  // Menú de Función de Activación
  printf("Elija su función de activación:\n");
  printf("1) RELU\n");
  printf("2) SIGMOIDE\n");
  printf("Opción: ");
  scanf("%d", &function_select);

  switch (function_select) {
  case 1:
    printf("-> Función RELU seleccionada.\n\n");
    break;
  case 2:
    printf("-> Función SIGMOIDE seleccionada.\n\n");
    break;
  default:
    printf("-> Opción no válida. Usando RELU por defecto.\n\n");
    function_select = 1;
  }

  // Menú de Inicialización de Pesos
  printf("Elija los pesos para la red:\n");
  printf("1) He Uniforme\n");
  printf("2) He Normal\n");
  printf("3) Xavier Uniforme\n");
  printf("4) Xavier Normal\n");
  printf("Opción: ");
  scanf("%d", &weights_select);

  switch (weights_select) {
  case 1:
    printf("-> Inicialización Uniforme seleccionada.\n");
    NN_Layer_innit_he(neuralnet, 0, input_size, 1 << 6, batch_size);
    NN_Layer_innit_he(neuralnet, 1, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_he(neuralnet, 2, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_he(neuralnet, 3, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_he(neuralnet, 4, 1 << 6, out_size, batch_size);
    break;
  case 2:
    printf("-> Inicialización Normal seleccionada.\n");
    NN_Layer_innit_normal(neuralnet, 0, input_size, 1 << 6, batch_size);
    NN_Layer_innit_normal(neuralnet, 1, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_normal(neuralnet, 2, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_normal(neuralnet, 3, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_normal(neuralnet, 4, 1 << 6, out_size, batch_size);
    break;
  case 3:
    printf("-> Inicialización Xavier Uniforme seleccionada.\n");
    // NeuralNetwork, LayerIdx, Input_size, Output_size, Batch_size
    NN_Layer_innit_xavier(neuralnet, 0, input_size, 1 << 6, batch_size);
    NN_Layer_innit_xavier(neuralnet, 1, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier(neuralnet, 2, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier(neuralnet, 3, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier(neuralnet, 4, 1 << 6, out_size, batch_size);
    break;
  case 4:
    printf("-> Inicialización Xavier Uniforme seleccionada.\n");
    // NeuralNetwork, LayerIdx, Input_size, Output_size, Batch_size
    NN_Layer_innit_xavier_normal(neuralnet, 0, input_size, 1 << 6, batch_size);
    NN_Layer_innit_xavier_normal(neuralnet, 1, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier_normal(neuralnet, 2, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier_normal(neuralnet, 3, 1 << 6, 1 << 6, batch_size);
    NN_Layer_innit_xavier_normal(neuralnet, 4, 1 << 6, out_size, batch_size);
    break;
  default:
    printf("-> Opción no válida. Usando Uniforme por defecto.\n");
    weights_select = 1;
  }

  NN_grad_innit(neuralnet);
  double start = omp_get_wtime();

  /*Aca le doy batch size en como argumento y no uso el de cada capa,
  para asi evitar llamados a cada cache de cada capa dado que son la misma
  siempre y porque ya lo tenia asi antes :D*/

  // NeuralNetwork, Epochs, X_data, Y_data, Number_Of_Examples, Batch_size,
  // Learning_Rate,Number_Threads
  switch (function_select) {
  case 1:
    NN_loop_relu(neuralnet, epochs, X, Y, n_examples, batch_size, learning_rate,
                 n_threads);
    break;
  case 2:
    NN_loop_sigmoid_fast(neuralnet, epochs, X, Y, n_examples, batch_size,
                         learning_rate, n_threads);
    break;
  }
  /* recordar que dar las funciones como argumento nos quita la posibilidad de
  hacer static inline en las funciones tal vez hacer una super funcion que
  ocupe un arguemneto tipo : 0, 1 ,2 para elegir la funcion a querer es mas
  feo pero deberia ir mas rapido nomas queda probarlo y ponerle static inline
  a lo otro, waaaaaa*/

  double end = omp_get_wtime();
  printf("Esto tardo %lf\n", end - start);

  // float ejemplo = 0.1;

  // if (function_select == 1) {
  //   for (int i = 0; i < 10; i++) {
  //     NN_prediction_relu(neuralnet, &ejemplo, 1);
  //     ejemplo += 0.1;
  //   }
  // } else if (function_select == 2) {
  //   for (int i = 0; i < 10; i++) {
  //     NN_prediction_sigmoid_fast(neuralnet, &ejemplo, 1);
  //     ejemplo += 0.1;
  //   }
  // }

  return 0;
}
