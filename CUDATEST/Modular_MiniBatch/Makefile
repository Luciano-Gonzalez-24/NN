CFLAGS = -Ofast -fopenmp -lm
NVCCFLAGS = -O3 -lineinfo -arch=native
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcublas -ldl

CC = gcc
NVCC = nvcc
RM = rm -f

EXEC = neural_network
EST = estructuras
TRAIN_CU = entrenamiento
PRED = prediccion

.PHONY: all run profile-nsys profile-ncu clean

all: $(EXEC)

$(EXEC): main.o $(EST).o $(TRAIN_CU).o $(PRED).o
	$(CC) $(CFLAGS) main.o $(EST).o $(TRAIN_CU).o $(PRED).o $(LDFLAGS) -o $(EXEC)

main.o: main.c $(EST).h $(TRAIN_CU)_cuda.h $(PRED).h
	$(CC) $(CFLAGS) -c main.c

$(EST).o: $(EST).c $(EST).h
	$(CC) $(CFLAGS) -c $(EST).c

$(TRAIN_CU).o: $(TRAIN_CU).cu $(TRAIN_CU)_cuda.h $(EST).h
	$(NVCC) $(NVCCFLAGS) -c $(TRAIN_CU).cu -o $(TRAIN_CU).o

$(PRED).o: $(PRED).c $(PRED).h $(EST).h
	$(CC) $(CFLAGS) -c $(PRED).c

run: $(EXEC)
	./$(EXEC)

profile-nsys: $(EXEC)
	nsys profile \
		--trace=cuda,nvtx,cublas \
		--stats=true \
		--force-overwrite=true \
		-o nn_custom_timeline \
		./$(EXEC)

profile-ncu: $(EXEC)
	ncu --set full \
		--kernel-name-base demangled \
		--kernel-name regex:"(forward_layer|add_bias_and_activate|output_delta|backward_delta|apply_act_deriv|grad_weights|grad_bias|sgd_update)_kernel" \
		--launch-count 12 \
		--force-overwrite \
		-o nn_custom_kernels \
		./$(EXEC)

clean:
	$(RM) main.o $(EST).o $(TRAIN_CU).o $(PRED).o $(EXEC) *.nsys-rep *.ncu-rep *.sqlite