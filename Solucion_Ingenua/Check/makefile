all: generate_random compare

generate_random: $(SRC_FILES) generate_random.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

compare: $(SRC_FILES) compare.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

clean:
	rm generate_random 2>/dev/null
	rm compare 2>/dev/null