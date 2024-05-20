all: generate_random compare

generate_random: $(SRC_FILES) generate_random.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

compare: $(SRC_FILES) compare.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)