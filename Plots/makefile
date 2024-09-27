
all: time_simulations_secuencial time_simulations_secuencial_vectorial time_simulations_OpenMP time_simulations_OpenMP_vectorial time_simulations_cuda

########################################################################
# Secuencial:
S_CCFLAGS = -DLINUX
S_LDFLAGS =
S_OPTFLAG = -O3

time_simulations_secuencial: time_simulations.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o 
	$(CC) $(S_LDFLAGS) -o time_simulations_secuencial \
	  time_simulations.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o -lm
	rm *.o

metodo_directo_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo.c -o metodo_directo_secuencial.o

metodo_directo_io_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo_io.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial.o

metodo_directo_grav_secuencial.o: ../Metodo_Directo/secuencial/src/metodo_directo_grav.c ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(S_OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial.o

########################################################################
# Secuencial:
S_CCFLAGS = -DLINUX
S_LDFLAGS =
S_OPTFLAG = -O3

time_simulations_secuencial_vectorial: time_simulations.c metodo_directo_secuencial_vectorial.o metodo_directo_io_secuencial_vectorial.o metodo_directo_grav_secuencial_vectorial.o 
	$(CC) $(S_LDFLAGS) -o time_simulations_secuencial_vectorial \
	  time_simulations.c metodo_directo_secuencial_vectorial.o metodo_directo_io_secuencial_vectorial.o metodo_directo_grav_secuencial_vectorial.o -lm
	rm *.o

metodo_directo_secuencial_vectorial.o: ../Metodo_Directo/secuencial/src/metodo_directo.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo.c -o metodo_directo_secuencial_vectorial.o

metodo_directo_io_secuencial_vectorial.o: ../Metodo_Directo/secuencial/src/metodo_directo_io.c ../Metodo_Directo/secuencial/inc/metodo_directo.h ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial_vectorial.o

metodo_directo_grav_secuencial_vectorial.o: ../Metodo_Directo/secuencial/src/metodo_directo_grav.c ../Metodo_Directo/secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(S_OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial_vectorial.o

########################################################################

# OpenMP:
O_CCFLAGS = -DLINUX
O_LDFLAGS = -lm -pthread
O_OPTFLAG = -O3

time_simulations_OpenMP: time_simulations.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o 
	$(CC) $(O_LDFLAGS) -o time_simulations_OpenMP \
	  time_simulations.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o -lm
	rm *.o

metodo_directo_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo.c -o metodo_directo_openmp.o

metodo_directo_io_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo_io.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp.o

metodo_directo_grav_openmp.o: ../Metodo_Directo/openmp/src/metodo_directo_grav.c ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(O_OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp.o

########################################################################

# OpenMP vectorial:
O_CCFLAGS = -DLINUX
O_LDFLAGS = -lm -pthread
O_OPTFLAG = -O3 -march=native

time_simulations_OpenMP_vectorial: time_simulations.c metodo_directo_openmp_vectorial.o metodo_directo_io_openmp_vectorial.o metodo_directo_grav_openmp_vectorial.o 
	$(CC) $(O_LDFLAGS) -o time_simulations_OpenMP_vectorial \
	  time_simulations.c metodo_directo_openmp_vectorial.o metodo_directo_io_openmp_vectorial.o metodo_directo_grav_openmp_vectorial.o -lm
	rm *.o

metodo_directo_openmp_vectorial.o: ../Metodo_Directo/openmp/src/metodo_directo.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo.c -o metodo_directo_openmp_vectorial.o

metodo_directo_io_openmp_vectorial.o: ../Metodo_Directo/openmp/src/metodo_directo_io.c ../Metodo_Directo/openmp/inc/metodo_directo.h ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp_vectorial.o

metodo_directo_grav_openmp_vectorial.o: ../Metodo_Directo/openmp/src/metodo_directo_grav.c ../Metodo_Directo/openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(O_OPTFLAG) $(DIAGNOSTICS) -c ../Metodo_Directo/openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp_vectorial.o

########################################################################

# cuda:
CUDA_CC:=nvcc 
LDFLAGS_CUDA = -lm
OPTFLAG_CUDA = -O3

time_simulations_cuda: time_simulations_cuda.cpp metodo_directo_cuda.o metodo_directo_io_cuda.o metodo_directo_grav_cuda.o 
	$(CUDA_CC) $(LDFLAGS_CUDA) -o time_simulations_cuda \
	  time_simulations_cuda.cpp metodo_directo_cuda.o metodo_directo_io_cuda.o metodo_directo_grav_cuda.o -lm
	rm *.o

metodo_directo_cuda.o: ../Metodo_Directo/cuda/src/metodo_directo.cu ../Metodo_Directo/cuda/inc/metodo_directo.h ../Metodo_Directo/cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(DIAGNOSTICS) -c ../Metodo_Directo/cuda/src/metodo_directo.cu -o metodo_directo_cuda.o

metodo_directo_io_cuda.o: ../Metodo_Directo/cuda/src/metodo_directo_io.cu ../Metodo_Directo/cuda/inc/metodo_directo.h ../Metodo_Directo/cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(DIAGNOSTICS) -c ../Metodo_Directo/cuda/src/metodo_directo_io.cu -o metodo_directo_io_cuda.o

metodo_directo_grav_cuda.o: ../Metodo_Directo/cuda/src/metodo_directo_grav.cu ../Metodo_Directo/cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(OPTFLAG_CUDA) $(DIAGNOSTICS) -c ../Metodo_Directo/cuda/src/metodo_directo_grav.cu -o metodo_directo_grav_cuda.o

########################################################################

clean: 
	rm -f time_simulations_secuencial
	rm -f time_simulations_OpenMP
	rm -f time_simulations_cuda
	rm -f time_simulations_secuencial_vectorial
	rm -f time_simulations_OpenMP_vectorial
	rm -f times.log
	rm -f fit.log
	rm *.o
	

