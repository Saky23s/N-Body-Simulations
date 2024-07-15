CC:=gcc
CFLAGS+=-std=c11
LDLIBS+=-lm

all: csv_to_binary graphic_starting_position sphere_starting_configuration galaxy_style_configuration

csv_to_binary: csv_to_binary.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

graphic_starting_position: graphic_starting_position.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

sphere_starting_configuration: sphere_starting_configuration.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

galaxy_style_configuration: galaxy_style_configuration.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

clean:
	rm galaxy_style_configuration 2>/dev/null
	rm sphere_starting_configuration 2>/dev/null
	rm graphic_starting_position 2>/dev/null
	rm csv_to_binary 2>/dev/null