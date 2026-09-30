#include "activation.h"
#include <math.h>
#include <stdlib.h>

float Relu(float input) { return (input > 0) * input; }

float Relu_prime(float input) { return (input > 0); }

float Sigmoid(float input) { 
  return (1 / (1 + powf(EULER_NUMBER, -input))); 
}

float Sigmoid_prime(float input) {
  return Sigmoid(input) * (1 - Sigmoid(input));
}

float Sigmoid_fast(float input) {
  return 0.5 *
         (input / 1 + (input > 0) * input + (input <= 0) * (-1.0) * input);
}

float Sigmoid_fast_prime(float input) {
  return Sigmoid_fast(input) * (1 - Sigmoid_fast(input));
}