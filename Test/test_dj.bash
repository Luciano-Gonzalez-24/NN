#!/bin/bash

# Compile the C program
echo "Compiling C program with GCC OpenMP..."
gcc -Wall -lm -fopenmp random_gen.c -o random_gen
echo -e "\n=========================================="
echo "Running Benchmarks for Different Thread Counts"
echo "=========================================="

# Test across different thread counts
for threads in 1 2 4 8 16
do
    echo "Running with OMP_NUM_THREADS=$threads..."
    export OMP_NUM_THREADS=$threads
    ./random_gen < input
    echo "------------------------------------------"
done

echo "=========================================="