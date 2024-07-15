CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

CUDA_CC:=nvcc
CUDA_FLAGS+=
CUDA_LDLIBS+=-lm 

SRC_FILES:=src/main.c 
CUDA_SRC_FILES:=src/main.cpp

all: simulation_secuencial simulation_OpenMP simulation_cuda

simulation_secuencial: $(SRC_FILES) src/simulation_secuencial.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

simulation_OpenMP: $(SRC_FILES) src/simulation_OpenMP.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

simulation_cuda: $(CUDA_SRC_FILES) src/simulation_cuda.cu 
	$(CUDA_CC) $^ $(CUDA_FLAGS) $(CUDA_LDLIBS) -o $@ $(CUDA_LDLIBS)

clean:
	rm simulation_OpenMP 2>/dev/null
	rm simulation_secuencial 2>/dev/null
	rm simulation_cuda 2>/dev/null