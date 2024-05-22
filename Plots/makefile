CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

CUDA_CC:=nvcc
CUDA_FLAGS+=
CUDA_LDLIBS+=-lm 

all: time_simulations time_simulations_cuda

time_simulations: time_simulations.c ../Solucion_Ingenua/src/simulation_secuencial.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_OpenMP: time_simulations.c ../Solucion_Ingenua/src/simulation_OpenMP.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_cuda: time_simulations_cuda.cpp ../Solucion_Ingenua/src/simulation_cuda.cu
	$(CUDA_CC) $^ $(CUDA_FLAGS) $(CUDA_LDLIBS) -o $@ $(CUDA_LDLIBS)

clean: 
	rm -f time_simulations
	rm -f time_simulations_cuda
	rm -f times.log
	rm -f fit.log
	

