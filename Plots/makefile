CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

CUDA_CC:=nvcc
CUDA_FLAGS+=
CUDA_LDLIBS+=-lm 

all: time_simulations_secuencial time_simulations_OpenMP time_simulations_secuencial_new time_simulations_openmp_new

time_simulations_secuencial: time_simulations.c ../Metodo_Directo/src/simulation_secuencial.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_OpenMP: time_simulations.c ../Metodo_Directo/src/simulation_OpenMP.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

time_simulations_cuda: time_simulations_cuda.cpp ../Metodo_Directo/src/simulation_cuda_testing_stuff.cu
	$(CUDA_CC) $^ $(CUDA_FLAGS) $(CUDA_LDLIBS) -o $@ $(CUDA_LDLIBS)

########################################################################
# Secuencial:
CCFLAGS = -DLINUX
LDFLAGS =
OPTFLAG = -O3

time_simulations_secuencial_new: time_simulations_new.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o 
	$(CC) $(LDFLAGS) -o time_simulations_secuencial_new \
	  time_simulations_new.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o -lm
	rm *.o

metodo_directo_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo.c -o metodo_directo_secuencial.o

metodo_directo_io_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo_io.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial.o

metodo_directo_grav_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo_grav.c ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial.o

########################################################################
# OpenMP:
CCFLAGS = -DLINUX
LDFLAGS = -lm -pthread
OPTFLAG = -O3

time_simulations_openmp_new: time_simulations_new.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o 
	$(CC) $(LDFLAGS) -o time_simulations_openmp_new \
	  time_simulations_new.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o -lm
	rm *.o

metodo_directo_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo.c -o metodo_directo_openmp.o

metodo_directo_io_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo_io.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp.o

metodo_directo_grav_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo_grav.c ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp.o



clean: 
	rm -f time_simulations_secuencial
	rm -f time_simulations_OpenMP
	rm -f time_simulations_cuda
	rm -f times.log
	rm -f fit.log
	

