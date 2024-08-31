##############################################################################
# MAKEFILE: compile and link treecode executable files                       #
# Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.                 #
##############################################################################

########################################################################
# DIRECTIONS: to tune this Makefile to your system, edit the definitions
# of CCFLAGS, LDFLAGS, OPTFLAG, and PREC below.  Suggested values
# are provided for LINUX, Sun, and SGI systems.

########################################################################
# Compiler options.

# LINUX:
CCFLAGS = -DLINUX
LDFLAGS =
OPTFLAG = -O3

# Uncomment the next line to see diagnostics instead of progress:
#DIAGNOSTICS = -DDIAGNOSTICS

all: treecode time_treecode_secuencial
########################################################################
# Vanila treecode:

treecode: src/main.c treecode.o treeio.o treeload.o treegrav.o libZeno.a
	$(CC) $(LDFLAGS) -o treecode \
	  src/main.c treecode.o treeio.o treeload.o treegrav.o libZeno.a -lm
	
time_treecode_secuencial: src/time.c treecode.o treeio.o treeload.o treegrav.o libZeno.a
	$(CC) $(LDFLAGS) -o time_treecode_secuencial \
	  src/time.c treecode.o treeio.o treeload.o treegrav.o libZeno.a -lm

treecode.o: src/treecode.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c src/treecode.c

treeio.o: src/treeio.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c src/treeio.c

treeload.o: src/treeload.c inc/treedefs.h
	$(CC) $(CCFLAGS) $(DIAGNOSTICS) -c src/treeload.c

treegrav.o: src/treegrav.c inc/treedefs.h
	$(CC) $(CCFLAGS) $(OPTFLAG) $(DIAGNOSTICS) -c src/treegrav.c

########################################################################
# Zeno library:

libZeno.a: clib.o mathfns.o
	ar rv libZeno.a clib.o mathfns.o

clib.o: src/clib.c inc/stdinc.h
	$(CC) $(CCFLAGS) -c src/clib.c

mathfns.o: src/mathfns.c inc/stdinc.h inc/mathfns.h
	$(CC) $(CCFLAGS) -c src/mathfns.c

########################################################################
clean:
	rm *.o 
	rm *.a 
	rm treecode
	rm time_treecode_secuencial

