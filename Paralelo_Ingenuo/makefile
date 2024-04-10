CC:=gcc
CFLAGS+=-std=c11 -fopenmp
LDLIBS+=-lm -pthread

SRC_FILES:=src/main.c src/simulation_OpenMP.c
main: $(SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)
