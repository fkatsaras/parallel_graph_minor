# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -march=native -ffast-math #-pg

# Object files
OBJS = serial_mul.o pthread_mul.o openmp_mul.o serial_gm.o pthread_gm.o openmp_gm.o mmio.o

# Target executable names
SERIAL_MUL_EXEC = serial_mul.exe
PTHREAD_MUL_EXEC = pthread_mul.exe
OPENMP_MUL_EXEC = openmp_mul.exe
SERIAL_GRAPHMINOR_EXEC = serial_gm.exe
PTHREAD_GRAPHMINOR_EXEC = pthread_gm.exe
OPENMP_GRAPHMINOR_EXEC = openmp_gm.exe

# -pg goes into the linking steps as wel

# Rule to build the serial matrix multiplication executable
$(SERIAL_MUL_EXEC): serial_mul.o mmio.o
	$(CC) -o $@ $^ -lm

# Rule to build the pthreads matrix multiplication executable
$(PTHREAD_MUL_EXEC): pthread_mul.o mmio.o
	$(CC) -pthread -o $@ $^ -lm 

# Rule to build the openmp matrix multiplication executable
$(OPENMP_MUL_EXEC): openmp_mul.o mmio.o
	$(CC) -fopenmp -o $@ $^ -lm

# Rule to build the serial graphminor executable
$(SERIAL_GRAPHMINOR_EXEC): serial_gm.o mmio.o
	$(CC) -o $@ $^ -lm

# Rule to build the pthreads graphminor executable
$(PTHREAD_GRAPHMINOR_EXEC): pthread_gm.o mmio.o
	$(CC) -pthread -o $@ $^ -lm

# Rule to build the openmp graphminor executable
$(OPENMP_GRAPHMINOR_EXEC): openmp_gm.o mmio.o
	$(CC) -fopenmp -o $@ $^ -lm

# General rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(SERIAL_MUL_EXEC) $(PTHREAD_MUL_EXEC) $(OPENMP_MUL_EXEC) $(SERIAL_GRAPHMINOR_EXEC) $(PTHREAD_GRAPHMINOR_EXEC) $(OPENMP_GRAPHMINOR_EXEC)
