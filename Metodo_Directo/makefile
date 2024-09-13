CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

CUDA_CC:=nvcc 
CUDA_FLAGS+=
CUDA_LDLIBS+=-lm 

SRC_FILES:=src/main.c 
CUDA_SRC_FILES:=src/main.cpp

all: simulation_secuencial simulation_OpenMP simulation_secuencial_new simulation_OpenMP_new

simulation_secuencial: $(SRC_FILES) src/simulation_secuencial.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

########################################################################
# Secuencial:
CCFLAGS = -DLINUX
LDFLAGS =
OPTFLAG = -O3

simulation_secuencial_new: secuencial/main.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o 
	$(CC) $(LDFLAGS) -o simulation_secuencial_new \
	  secuencial/main.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o -lm
	rm *.o

metodo_directo_secuencial.o: secuencial/src/metodo_directo.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo.c -o metodo_directo_secuencial.o

metodo_directo_io_secuencial.o: secuencial/src/metodo_directo_io.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial.o

metodo_directo_grav_secuencial.o: secuencial/src/metodo_directo_grav.c secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial.o

########################################################################

# OpenMP:
CCFLAGS = -DLINUX
LDFLAGS = -lm -pthread
OPTFLAG = -O3

simulation_OpenMP_new: openmp/main.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o 
	$(CC) $(LDFLAGS) -o simulation_OpenMP_new \
	  openmp/main.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o -lm
	rm *.o

metodo_directo_openmp.o: openmp/src/metodo_directo.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c openmp/src/metodo_directo.c -o metodo_directo_openmp.o

metodo_directo_io_openmp.o: openmp/src/metodo_directo_io.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp.o

metodo_directo_grav_openmp.o: openmp/src/metodo_directo_grav.c openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) $(DIAGNOSTICS) -c openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp.o

########################################################################

simulation_OpenMP: $(SRC_FILES) src/simulation_OpenMP.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

simulation_cuda: $(CUDA_SRC_FILES) src/simulation_cuda.cu
	$(CUDA_CC) $^ $(CUDA_FLAGS) $(CUDA_LDLIBS) -o $@ $(CUDA_LDLIBS)

clean:
	rm simulation_OpenMP 2>/dev/null
	rm simulation_secuencial 2>/dev/null
	rm simulation_secuencial_new 2>/dev/null
	rm simulation_cuda 2>/dev/null
	rm *.o

	
