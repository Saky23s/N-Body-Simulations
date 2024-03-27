CC:=gcc
CFLAGS+=-std=c11
LDLIBS+=-lm

SRC_FILES:=src/main.c src/simulation.c
main: $(SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)
