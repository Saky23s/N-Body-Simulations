all: compare 

compare: $(SRC_FILES) compare.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@ $(LDLIBS)

clean:
	rm compare 2>/dev/null