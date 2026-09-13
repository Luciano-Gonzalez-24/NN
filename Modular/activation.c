#include "activation.h"
#include <math.h>

float Relu(float input) { return (input > 0) * input; }

float Relu_prime(float input) { return (input > 0) * 1.0; }

float Sigmoid(float input) { return (1 / (1 + pow(EULER_NUMBER, -input))); }

float Sigmoid_prime(float input) {
  return Sigmoid(input) * (1 - Sigmoid(input));
}