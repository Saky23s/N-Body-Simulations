
all: simulation_secuencial simulation_secuencial_vectorial simulation_OpenMP simulation_OpenMP_vectorial simulation_cuda

#UNCOMMENT THIS LINE FOR DIAGNOSTICS
#DIAGNOSTICS = -DDIAGNOSTICS

########################################################################
# Secuencial:
S_CCFLAGS = -DLINUX
S_LDFLAGS =
S_OPTFLAG = -O3

simulation_secuencial: main.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o 
	$(CC) $(S_LDFLAGS) -o simulation_secuencial \
	  main.c metodo_directo_secuencial.o metodo_directo_io_secuencial.o metodo_directo_grav_secuencial.o -lm
	rm *.o

metodo_directo_secuencial.o: secuencial/src/metodo_directo.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo.c -o metodo_directo_secuencial.o

metodo_directo_io_secuencial.o: secuencial/src/metodo_directo_io.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial.o

metodo_directo_grav_secuencial.o: secuencial/src/metodo_directo_grav.c secuencial/inc/medoto_directo_defs.h
	$(CC) $(S_CCFLAGS) $(S_OPTFLAG) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial.o

########################################################################
# Secuencial vectorial:
SV_CCFLAGS = -DLINUX 
SV_OPTALLFLAGS = -march=native
SV_LDFLAGS =
SV_OPTFLAG = -O3 

simulation_secuencial_vectorial: main.c metodo_directo_secuencial_vectorial.o metodo_directo_io_secuencial_vectorial.o metodo_directo_grav_secuencial_vectorial.o 
	$(CC) $(SV_LDFLAGS) -o simulation_secuencial_vectorial \
	  main.c metodo_directo_secuencial_vectorial.o metodo_directo_io_secuencial_vectorial.o metodo_directo_grav_secuencial_vectorial.o -lm
	rm *.o

metodo_directo_secuencial_vectorial.o: secuencial/src/metodo_directo.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(SV_CCFLAGS) $(SV_OPTALLFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo.c -o metodo_directo_secuencial_vectorial.o

metodo_directo_io_secuencial_vectorial.o: secuencial/src/metodo_directo_io.c secuencial/inc/metodo_directo.h secuencial/inc/medoto_directo_defs.h
	$(CC) $(SV_CCFLAGS) $(SV_OPTALLFLAGS) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_io.c -o metodo_directo_io_secuencial_vectorial.o

metodo_directo_grav_secuencial_vectorial.o: secuencial/src/metodo_directo_grav.c secuencial/inc/medoto_directo_defs.h
	$(CC) $(SV_CCFLAGS) $(SV_OPTALLFLAGS) $(SV_OPTFLAG) $(DIAGNOSTICS) -c secuencial/src/metodo_directo_grav.c -o metodo_directo_grav_secuencial_vectorial.o
########################################################################

# OpenMP:
O_CCFLAGS = -DLINUX
O_LDFLAGS = -lm -lpthread
FOPENMP=-fopenmp
O_OPTFLAG = -O3

simulation_OpenMP: main.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o 
	$(CC) $(FOPENMP) $(O_LDFLAGS) -o simulation_OpenMP \
	  main.c metodo_directo_openmp.o metodo_directo_io_openmp.o metodo_directo_grav_openmp.o -lm
	rm *.o

metodo_directo_openmp.o: openmp/src/metodo_directo.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(FOPENMP) $(DIAGNOSTICS) -c openmp/src/metodo_directo.c -o metodo_directo_openmp.o

metodo_directo_io_openmp.o: openmp/src/metodo_directo_io.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(FOPENMP) $(DIAGNOSTICS) -c openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp.o

metodo_directo_grav_openmp.o: openmp/src/metodo_directo_grav.c openmp/inc/medoto_directo_defs.h
	$(CC) $(O_CCFLAGS) $(FOPENMP) $(O_OPTFLAG) $(DIAGNOSTICS) -c openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp.o

########################################################################

# OpenMP vectorial:
OV_CCFLAGS = -DLINUX
OV_LDFLAGS = -lm 
OV_OPTALLFLAG = -march=native
OV_OPTFLAG = -O3 

simulation_OpenMP_vectorial: main.c metodo_directo_openmp_vectorial.o metodo_directo_io_openmp_vectorial.o metodo_directo_grav_openmp_vectorial.o 
	$(CC) $(FOPENMP) $(OV_LDFLAGS) -o simulation_OpenMP_vectorial \
	  main.c metodo_directo_openmp_vectorial.o metodo_directo_io_openmp_vectorial.o metodo_directo_grav_openmp_vectorial.o -lm
	rm *.o

metodo_directo_openmp_vectorial.o: openmp/src/metodo_directo.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(OV_CCFLAGS) $(FOPENMP) $(OV_OPTALLFLAG) $(DIAGNOSTICS) -c openmp/src/metodo_directo.c -o metodo_directo_openmp_vectorial.o

metodo_directo_io_openmp_vectorial.o: openmp/src/metodo_directo_io.c openmp/inc/metodo_directo.h openmp/inc/medoto_directo_defs.h
	$(CC) $(OV_CCFLAGS) $(FOPENMP) $(OV_OPTALLFLAG) $(DIAGNOSTICS) -c openmp/src/metodo_directo_io.c -o metodo_directo_io_openmp_vectorial.o

metodo_directo_grav_openmp_vectorial.o: openmp/src/metodo_directo_grav.c openmp/inc/medoto_directo_defs.h
	$(CC) $(OV_CCFLAGS) $(FOPENMP) $(OV_OPTALLFLAG) $(OV_OPTFLAG) $(DIAGNOSTICS) -c openmp/src/metodo_directo_grav.c -o metodo_directo_grav_openmp_vectorial.o

########################################################################

# cuda:
CUDA_CC:= nvcc 
CUDA_CCFLAGS = -DLINUX -w
LDFLAGS_CUDA = -lm
OPTFLAG_CUDA = -O3

simulation_cuda: main.cpp metodo_directo_cuda.o metodo_directo_io_cuda.o metodo_directo_grav_cuda.o 
	$(CUDA_CC) $(CUDA_CCFLAGS) $(LDFLAGS_CUDA) $(OPTFLAG_CUDA) -o simulation_cuda \
	  main.cpp metodo_directo_cuda.o metodo_directo_io_cuda.o metodo_directo_grav_cuda.o -lm
	rm *.o

metodo_directo_cuda.o: cuda/src/metodo_directo.cu cuda/inc/metodo_directo.h cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(CUDA_CCFLAGS) $(OPTFLAG_CUDA) $(DIAGNOSTICS) -c cuda/src/metodo_directo.cu -o metodo_directo_cuda.o

metodo_directo_io_cuda.o: cuda/src/metodo_directo_io.cu cuda/inc/metodo_directo.h cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(CUDA_CCFLAGS) $(OPTFLAG_CUDA) $(DIAGNOSTICS) -c cuda/src/metodo_directo_io.cu -o metodo_directo_io_cuda.o

metodo_directo_grav_cuda.o: cuda/src/metodo_directo_grav.cu cuda/inc/medoto_directo_defs.h
	$(CUDA_CC) $(CUDA_CCFLAGS) $(OPTFLAG_CUDA) $(DIAGNOSTICS) -c cuda/src/metodo_directo_grav.cu -o metodo_directo_grav_cuda.o

########################################################################


clean:
	rm simulation_secuencial 2>/dev/null
	rm simulation_OpenMP 2>/dev/null
	rm simulation_cuda 2>/dev/null
	rm simulation_secuencial_vectorial 2>/dev/null
	rm simulation_OpenMP_vectorial 2>/dev/null

	
