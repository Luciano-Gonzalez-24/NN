CC      = gcc
RM      = rm -f

# ---- MKL ----------------------------------------------------------------
# Ejecuta antes:  source /opt/intel/oneapi/setvars.sh   (define MKLROOT)
MKLROOT    ?= /opt/intel/oneapi/mkl/latest
MKL_LIBDIR := $(firstword $(wildcard $(MKLROOT)/lib/intel64 $(MKLROOT)/lib))

# ---- Flags --------------------------------------------------------------
OPT     ?= -Ofast
ARCH    ?=
CFLAGS  = $(OPT) $(ARCH) -fopenmp -I$(MKLROOT)/include
LDFLAGS = -L$(MKL_LIBDIR) -Wl,-rpath,$(MKL_LIBDIR) -Wl,--no-as-needed
LDLIBS  = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

# ---- Archivos -----------------------------------------------------------
EXEC      = neural_network
EST       = estructuras
TRAIN     = entrenamiento
PRED      = prediccion
# Fuente de entrenamiento a usar (la version MKL). Cambiala con:
#   make TRAIN_SRC=entrenamiento.c
TRAIN_SRC ?= entrenamiento_mkl.c

# ---- Ejecucion ----------------------------------------------------------
ARGS    ?=
THREADS ?= 1 2 4 8 12 16 24
PIN     = OMP_PROC_BIND=close OMP_PLACES=cores MKL_THREADING_LAYER=GNU

all: check $(EXEC)

check:
	@test -f $(MKLROOT)/include/mkl.h || (echo "ERROR: no encuentro mkl.h en '$(MKLROOT)/include'. Ejecuta: source /opt/intel/oneapi/setvars.sh" && exit 1)

$(EXEC): main.o $(EST).o $(TRAIN).o $(PRED).o
	$(CC) $(CFLAGS) main.o $(EST).o $(TRAIN).o $(PRED).o -o $(EXEC) $(LDFLAGS) $(LDLIBS)

main.o: main.c $(EST).h $(TRAIN).h $(PRED).h
	$(CC) $(CFLAGS) -c main.c

$(EST).o: $(EST).c $(EST).h
	$(CC) $(CFLAGS) -c $(EST).c

$(TRAIN).o: $(TRAIN_SRC) $(EST).h $(TRAIN).h
	$(CC) $(CFLAGS) -c $(TRAIN_SRC) -o $(TRAIN).o

$(PRED).o: $(PRED).c $(PRED).h $(EST).h
	$(CC) $(CFLAGS) -c $(PRED).c

# ---- Modos de ejecucion -------------------------------------------------
# Sin fijar hilos (como antes)
run:
	./$(EXEC) $(ARGS)

# Hilos fijados a nucleos fisicos, MKL con el mismo OpenMP (GNU)
run-pinned:
	$(PIN) ./$(EXEC) $(ARGS)

# Un solo socket (para 12 hilos o menos: sin trafico entre sockets)
run-socket0:
	$(PIN) numactl --cpunodebind=0 --membind=0 ./$(EXEC) $(ARGS)

# 24 hilos repartidos entre los dos sockets
run-spread:
	OMP_PROC_BIND=spread OMP_PLACES=cores MKL_THREADING_LAYER=GNU ./$(EXEC) $(ARGS)

# Barrido de hilos. Asume que el 1er argumento del programa es n_threads.
# Guarda las lineas CSV en resultados.csv (la version hibrida anade fw_max y bw_max;
# la version entrenamiento_mkl.c solo imprime las 5 primeras columnas)
bench:
	@echo "hilos,total,forward,backward,update,resto,fw_max,bw_max" > resultados.csv
	@for T in $(THREADS); do \
		echo "== $$T hilos"; \
		$(PIN) ./$(EXEC) $$T | grep "^CSV" | sed 's/^CSV,//' >> resultados.csv; \
	done
	@cat resultados.csv

clean:
	$(RM) main.o $(EST).o $(TRAIN).o $(PRED).o $(EXEC)

.PHONY: all check run run-pinned run-socket0 run-spread bench clean
