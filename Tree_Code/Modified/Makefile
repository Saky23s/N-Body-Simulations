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

# Sun:
# CCFLAGS =
# LDFLAGS =
# OPTFLAG = -xO3

# SGI:
# CCFLAGS = -n32
# LDFLAGS = -n32
# OPTFLAG = -O3

########################################################################
# Precision.  Possible values are SINGLEPREC, MIXEDPREC, and DOUBLEPREC.

# LINUX, SGI:
PREC = DOUBLEPREC

# Sun:
# PREC = MIXEDPREC

########################################################################
# Variations:

# Uncomment the next line to use freq instead of dtime:
# USEFREQ = -DUSEFREQ

# Uncomment the next line to use binary I/O:
# BINARYIO = -DBINARYIO

########################################################################
# Vanila treecode:

treecode: treecode.o treeio.o treeload.o treegrav.o libZeno.a
	$(CC) $(LDFLAGS) -o treecode \
	  treecode.o treeio.o treeload.o treegrav.o libZeno.a -lm

treecode.o: src/treecode.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) $(USEFREQ) -c src/treecode.c

treeio.o: src/treeio.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) $(USEFREQ) $(BINARYIO) -c src/treeio.c

treeload.o: src/treeload.c inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) -c src/treeload.c

treegrav.o: src/treegrav.c inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) -c src/treegrav.c

########################################################################
# Quick-scan treecode:

treecode_q: treecode_q.o treeio_q.o treeload_q.o treegrav_q.o libZeno.a
	$(CC) $(LDFLAGS) -o treecode_q \
	  treecode_q.o treeio_q.o treeload_q.o treegrav_q.o libZeno.a -lm

treecode_q.o: src/treecode.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) -DQUICKSCAN $(USEFREQ) -c \
	  -o treecode_q.o src/treecode.c

treeio_q.o: src/treeio.c inc/treecode.h inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) -DQUICKSCAN $(USEFREQ) $(BINARYIO) -c \
	  -o treeio_q.o src/treeio.c

treeload_q.o: src/treeload.c inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) -DQUICKSCAN -c \
	  -o treeload_q.o src/treeload.c

treegrav_q.o: src/treegrav.c inc/treedefs.h
	$(CC) $(CCFLAGS) -D$(PREC) -DQUICKSCAN $(OPTFLAG) -c \
	  -o treegrav_q.o src/treegrav.c

########################################################################
# Zeno library:

libZeno.a: clib.o getparam.o mathfns.o
	ar ruv libZeno.a clib.o getparam.o mathfns.o

clib.o: src/clib.c inc/stdinc.h
	$(CC) $(CCFLAGS) -D$(PREC) -c src/clib.c

getparam.o: src/getparam.c inc/stdinc.h inc/getparam.h
	$(CC) $(CCFLAGS) -D$(PREC) -c src/getparam.c

mathfns.o: src/mathfns.c inc/stdinc.h inc/mathfns.h
	$(CC) $(CCFLAGS) -D$(PREC) -c src/mathfns.c

########################################################################
clean:
	rm *.o 
	rm *.a 
	rm treecode

