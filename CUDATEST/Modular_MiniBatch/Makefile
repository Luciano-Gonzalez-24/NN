CFLAGS = -Ofast -fopenmp -lm
NVCCFLAGS = -O3 -arch=native

CC = gcc
NVCC = nvcc
RM = rm -f

EXEC = neural_network
EST = estructuras
TRAIN_CU = entrenamiento
PRED = prediccion

all: $(EXEC)

$(EXEC): main.o $(EST).o $(TRAIN_CU).o $(PRED).o
	$(CC) $(CFLAGS) main.o $(EST).o $(TRAIN_CU).o $(PRED).o -lcudart -L/usr/local/cuda/lib64 -o $(EXEC)

main.o: main.c $(EST).h $(TRAIN_CU)_cuda.h $(PRED).h
	$(CC) $(CFLAGS) -c main.c

$(EST).o: $(EST).c $(EST).h
	$(CC) $(CFLAGS) -c $(EST).c

$(TRAIN_CU).o: $(TRAIN_CU).cu $(TRAIN_CU)_cuda.h $(EST).h
	$(NVCC) $(NVCCFLAGS) -c $(TRAIN_CU).cu -o $(TRAIN_CU).o

$(PRED).o: $(PRED).c $(PRED).h $(EST).h
	$(CC) $(CFLAGS) -c $(PRED).c

run:
	./$(EXEC)

clean:
	$(RM) main.o $(EST).o $(TRAIN_CU).o $(PRED).o $(EXEC)