CC:=gcc
CFLAGS+=-std=c11
LDLIBS+=-lm

all: csv_to_binary graphic_starting_position

csv_to_binary: csv_to_binary.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

graphic_starting_position: graphic_starting_position.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)
