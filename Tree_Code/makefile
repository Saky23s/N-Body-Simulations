##############################################################################
# MAKEFILE: compile and link treecode executable files                       #
# Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.                 #
##############################################################################

########################################################################
# Compiler options.

# LINUX:
CCFLAGS = -DLINUX
LDFLAGS =
OPTFLAG = -O3

########################################################################
# Secuential treecode:
treecode_secuencial: src/main.c treecode.o treeload.o treegrav.o libZeno.a
	$(CC) $(LDFLAGS) -o treecode_secuencial \
	  src/main.c treecode.o treeload.o treegrav.o libZeno.a -lm

treecode.o: src/treecode.c inc/treedefs.h
	$(CC) $(CCFLAGS) -c src/treecode.c

treeload.o: src/treeload.c inc/treedefs.h
	$(CC) $(CCFLAGS) -c src/treeload.c

treegrav.o: src/treegrav.c inc/treedefs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) -c src/treegrav.c


########################################################################
# Zeno library:
libZeno.a: clib.o 
	ar ruv libZeno.a clib.o 

clib.o: src/clib.c inc/stdinc.h
	$(CC) $(CCFLAGS) -c src/clib.c


########################################################################
clean:
	rm *.o 
	rm libZeno.a
	rm treecode_secuencial