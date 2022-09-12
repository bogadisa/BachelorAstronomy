CC = mpicc
CFLAGS = -O2
LDFLAGS = -L../one_folder/simple-jpeg/ -lsimplejpeg
PROFILE := -pg

PROJ = parallel_main.exe
OBJS = parallel_main.c
DEPS = common.h functions.h

all: simple-jpeg $(PROJ)

parallel_main.exe: parallel_main.c $(DEPS)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

simple-jpeg:
	$(MAKE) -C ../one_folder/simple-jpeg

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean
	$(MAKE) -C ../one_folder/simple-jpeg clean