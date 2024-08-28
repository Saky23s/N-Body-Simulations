##############################################################################
# MAKEFILE: compile and link treecode executable files                       #
# author Santiago Salas santiago.salas@estudiante.uam.es					 #
##############################################################################

########################################################################
# Compiler options.

# LINUX:
CCFLAGS = -DLINUX
LDFLAGS =
OPTFLAG = -O3

########################################################################
# Secuential treecode:
treecode_secuencial: src/main.c treecode.o treeload.o treegrav.o
	$(CC) $(LDFLAGS) -o treecode_secuencial \
	  src/main.c treecode.o treeload.o treegrav.o -lm

treecode.o: src/treecode.c inc/treedefs.h
	$(CC) $(CCFLAGS) -c src/treecode.c

treeload.o: src/treeload.c inc/treedefs.h
	$(CC) $(CCFLAGS) -c src/treeload.c

treegrav.o: src/treegrav.c inc/treedefs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) -c src/treegrav.c


########################################################################
clean:
	rm *.o 
	rm treecode_secuencial