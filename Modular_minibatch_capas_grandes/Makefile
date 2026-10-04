CFLAGS= -Ofast -fopenmp -lm

CC=gcc

RM=rm -f

EXEC=neural_network
EST = estructuras
TRAIN = entrenamiento
ACT = activation
PRED = prediccion

all: $(EXEC)

$(EXEC): main.o $(EST).o $(TRAIN).o $(ACT).o $(PRED).o
	$(CC) $(CFLAGS) main.o $(EST).o $(TRAIN).o $(ACT).o $(PRED).o -o $(EXEC)

main.o: main.c $(EST).h $(TRAIN).h $(ACT).h $(PRED).o
	$(CC) $(CFLAGS) -c main.c

$(EST).o: $(EST).c $(EST).h
	$(CC) $(CFLAGS) -c $(EST).c

$(TRAIN).o: $(TRAIN).c $(EST).h $(TRAIN).h $(ACT).h
	$(CC) $(CFLAGS) -c $(TRAIN).c

$(PRED).o: $(PRED).c $(PRED).h $(EST).h $(ACT).h
	$(CC) $(CFLAGS) -c $(PRED).c


run:
	./$(EXEC)

clean:
	$(RM) main.o $(EST).o $(TRAIN).o $(ACT).o $(PRED).o  $(EXEC)
