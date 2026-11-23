#include "entrenamiento.h"
#include "estructuras.h"
#include <mkl.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Entrenamiento mini-batch con GEMM (Intel oneMKL, interfaz CBLAS).
 *
 * Convenciones (todo row-major):
 *   weights      : out x in
 *   cache.z      : B x out
 *   output       : B x out
 *   cache.grad   : B x out   (dL/da de ESA capa; luego se usa in-place como delta)
 *
 * Requisitos (los mismos que ya tenias):
 *   - cache.z y cache.grad reservados con batch_size * output_size en NN_Layer_innit
 *   - el batch_size de NN_Layer_innit == batch_size de NN_loop_*
 *   - num_layers >= 1
 *
 * Hilos: MKL usa sus propios hilos (n_threads). Las llamadas a sgemm se hacen
 * desde codigo secuencial; OpenMP solo se usa en los bucles elementales.
 *
 * Compilar (GCC, MKL con hilos GNU OpenMP, interfaz LP64):
 *   gcc -O2 -fopenmp entrenamiento_mkl.c ... -I${MKLROOT}/include \
 *       -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed \
 *       -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
 * (o usa el enlazador de Intel: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)
 */

#ifndef NN_VERBOSE
#define NN_VERBOSE 1
#endif

enum { ACT_RELU = 0, ACT_SIGMOID_FAST = 1 };

static inline float act_fn(int act, float z) {
  if (act == ACT_RELU)
    return (z > 0.0f) * z;
  return 0.5f * z / (1.0f + fabsf(z)) + 0.5f;
}

static inline float act_prime(int act, float z) {
  if (act == ACT_RELU)
    return (z > 0.0f) ? 1.0f : 0.0f;
  float m = 1.0f + fabsf(z);
  return 0.5f / (m * m);
}

static int NN_loop_gemm(NeuralNetwork *neuralnet, int epochs, float *X,
                        float *Y, int n_examples, int batch_size,
                        float learning_rate, int n_threads, int act) {

  const int L = neuralnet->num_layers;
  const int B = batch_size;
  const int final_size = neuralnet->layers[L - 1].output_size;
  const int n_batches = n_examples / B; // los ejemplos sobrantes se descartan
  const float lr_over_batch = learning_rate / (float)B;

  mkl_set_dynamic(1);              // respeta exactamente n_threads
  mkl_set_num_threads(n_threads);

  // Buffers de salida (B x out) y gradientes de parametros (un solo juego)
  float **gW = (float **)malloc(L * sizeof(float *));
  float **gB = (float **)malloc(L * sizeof(float *));
  if (gW == NULL || gB == NULL) {
    printf("No se pudo reservar memoria\n");
    return -1;
  }
  for (int l = 0; l < L; l++) {
    DenseLayer *Ly = &neuralnet->layers[l];
    size_t nw = (size_t)Ly->input_size * Ly->output_size;
    Ly->output = (float *)calloc((size_t)B * Ly->output_size, sizeof(float));
    gW[l] = (float *)malloc(nw * sizeof(float));
    gB[l] = (float *)malloc((size_t)Ly->output_size * sizeof(float));
    if (Ly->output == NULL || gW[l] == NULL || gB[l] == NULL) {
      printf("No se pudo reservar memoria en la capa %d\n", l);
      return -1;
    }
  }

  for (int ep = 0; ep < epochs; ep++) {
    float loss_avg = 0.0f;

    for (int batch = 0; batch < n_batches; batch++) {

      /* ===================== FORWARD: Z = A * W^T ===================== */
      for (int l = 0; l < L; l++) {
        DenseLayer *Ly = &neuralnet->layers[l];
        const int in = Ly->input_size;
        const int out = Ly->output_size;
        const float *A = (l == 0) ? X + (size_t)batch * B * in
                                  : neuralnet->layers[l - 1].output;
        float *Z = Ly->cache.z;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, out, in, 1.0f,
                    A, in, Ly->weights, in, 0.0f, Z, out);

        // bias + activacion (ultima capa: lineal)
        const int is_last = (l == L - 1);
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int b = 0; b < B; b++) {
          float *zr = Z + (size_t)b * out;
          float *orow = Ly->output + (size_t)b * out;
          for (int i = 0; i < out; i++) {
            float z = zr[i] + Ly->bias[i];
            zr[i] = z;
            orow[i] = is_last ? z : act_fn(act, z);
          }
        }
      }

      /* ===================== SEMILLA: dL/da = salida - y ===================== */
      {
        DenseLayer *Last = &neuralnet->layers[L - 1];
        const float *Yb = Y + (size_t)batch * B * final_size;
        float *g = Last->cache.grad;
        const float *o = Last->output;
        const long total = (long)B * final_size;
        float loss_local = 0.0f;
#pragma omp parallel for num_threads(n_threads) reduction(+ : loss_local)
        for (long k = 0; k < total; k++) {
          float diff = o[k] - Yb[k];
          g[k] = diff;
          loss_local += 0.5f * diff * diff;
        }
        loss_avg += loss_local / (float)n_examples / (float)final_size;
      }

      /* ===================== BACKWARD ===================== */
      for (int l = L - 1; l >= 0; l--) {
        DenseLayer *Ly = &neuralnet->layers[l];
        const int in = Ly->input_size;
        const int out = Ly->output_size;
        const float *A = (l == 0) ? X + (size_t)batch * B * in
                                  : neuralnet->layers[l - 1].output;
        float *delta = Ly->cache.grad; // B x out

        // delta = dL/da * f'(z)   (ultima capa lineal: f' = 1)
        if (l < L - 1) {
          const float *Z = Ly->cache.z;
          const long total = (long)B * out;
#pragma omp parallel for num_threads(n_threads) schedule(static)
          for (long k = 0; k < total; k++)
            delta[k] *= act_prime(act, Z[k]);
        }

        // gB[i] = suma sobre el batch de delta[b][i]
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int i = 0; i < out; i++) {
          float s = 0.0f;
          for (int b = 0; b < B; b++)
            s += delta[(size_t)b * out + i];
          gB[l][i] = s;
        }

        // gW = delta^T * A      (out x in)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out, in, B, 1.0f,
                    delta, out, A, in, 0.0f, gW[l], in);

        // dA_prev = delta * W   (B x in)  -> usa W viejo (se actualiza despues)
        if (l > 0) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B, in, out,
                      1.0f, delta, out, Ly->weights, in, 0.0f,
                      neuralnet->layers[l - 1].cache.grad, in);
        }
      }

      /* ===================== UPDATE ===================== */
      for (int l = 0; l < L; l++) {
        DenseLayer *Ly = &neuralnet->layers[l];
        const int in = Ly->input_size;
        const int out = Ly->output_size;

        cblas_saxpy(in * out, -lr_over_batch, gW[l], 1, Ly->weights, 1);
        cblas_saxpy(out, -lr_over_batch, gB[l], 1, Ly->bias, 1);
      }
    } // batches

#if NN_VERBOSE
    printf("En la epoca %d, el loss es: %lf \n", ep, (double)loss_avg);
#endif
  } // epocas

  for (int l = 0; l < L; l++) {
   // free(neuralnet->layers[l].output);
   // neuralnet->layers[l].output = NULL;
    free(gW[l]);
    free(gB[l]);
  }
  free(gW);
  free(gB);

  return 0;
}

int NN_loop_relu(NeuralNetwork *neuralnet, int epochs, float *X, float *Y,
                 int n_examples, int batch_size, float learning_rate,
                 int n_threads) {
  return NN_loop_gemm(neuralnet, epochs, X, Y, n_examples, batch_size,
                      learning_rate, n_threads, ACT_RELU);
}

int NN_loop_sigmoid_fast(NeuralNetwork *neuralnet, int epochs, float *X,
                         float *Y, int n_examples, int batch_size,
                         float learning_rate, int n_threads) {
  return NN_loop_gemm(neuralnet, epochs, X, Y, n_examples, batch_size,
                      learning_rate, n_threads, ACT_SIGMOID_FAST);
}
