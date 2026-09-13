CFLAGS=-O3 -fopenmp

CC=gcc

RM=rm -f

EXEC=neural_network
EST = estructuras
TRAIN = entrenamiento

all: $(EXEC)

$(EXEC): main.o $(EST).o $(TRAIN).o
	$(CC) $(CFLAGS) main.o $(EST).o $(TRAIN).o -o $(EXEC)

main.o: main.c $(EST).h $(TRAIN).h
	$(CC) $(CFLAGS) -c main.c

$(EST).o: $(EST).c $(EST).h
	$(CC) $(CFLAGS) -c $(EST).c

$(TRAIN).o: $(TRAIN).c $(EST).h $(TRAIN).h
	$(CC) $(CFLAGS) -c $(TRAIN).c

run:
	./$(EXEC)

clean:
	$(RM) main.o $(EST).o $(TRAIN).o $(EXEC)
