CC:=gcc
CFLAGS+=-std=c11
LDLIBS+=-lm

time_simulations: time_simulations.c ../Secuencial_Ingenuo/src/simulation.c 
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

