CC := gcc
PROFILE := -pg

PROGRAMS = $(patsubst %.c,%.exe,$(wildcard *.c)) read_graph_from_file.exe # PageRank_iterations.exe top_n_webpages.exe PageRank_iterations_omp.exe main.exe
DEPS = read_graph_from_file.h

.PHONY : all clean

all: $(PROGRAMS)

read_graph_from_file.exe : read_graph_from_file.c $(DEPS)
	$(CC) $< -o $@

PageRank_iterations.exe : PageRank_iterations.c $(PageRank_iterations.h read_graph_from_file.h)
	$(CC) $< -o $@

top_n_webpages.exe : top_n_webpages.c $(top_n_webpages.h PageRank_iterations.h read_graph_from_file.h)
	$(CC) $< -o $@

PageRank_iterations_omp.exe : PageRank_iterations_omp.c $(PageRank_iterations_omp.h read_graph_from_file.h)
	$(CC) $< -o $@ -fopenmp

main.exe : main.c $(top_n_webpages.h PageRank_iterations.h read_graph_from_file.h)
	$(CC) $< -o $@ 


clean :
	rm -f *.o $(PROGRAMS)

print-%  : ; @echo $* = $($*)