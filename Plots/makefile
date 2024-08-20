CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

CUDA_CC:=nvcc
CUDA_FLAGS+=
CUDA_LDLIBS+=-lm 

all: time_simulations_secuencial time_simulations_OpenMP time_simulations_cuda

time_simulations_secuencial: time_simulations.c ../Metodo_Directo/src/simulation_secuencial.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_OpenMP: time_simulations.c ../Metodo_Directo/src/simulation_OpenMP.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_cuda: time_simulations_cuda.cpp ../Metodo_Directo/src/simulation_cuda_testing_stuff.cu
	$(CUDA_CC) $^ $(CUDA_FLAGS) $(CUDA_LDLIBS) -o $@ $(CUDA_LDLIBS)

clean: 
	rm -f time_simulations_secuencial
	rm -f time_simulations_OpenMP
	rm -f time_simulations_cuda
	rm -f times.log
	rm -f fit.log
	

