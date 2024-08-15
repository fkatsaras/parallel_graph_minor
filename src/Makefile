# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -march=native -ffast-math

# Object files
OBJS = serial_mul.o pthread_mul.o openmp_mul.o serial_gm.o pthread_gm.o openmp_gm.o mmio.o

# Target executable names
SERIAL_MUL_EXEC = serial_mul
PTHREAD_MUL_EXEC = pthread_mul
OPENMP_MUL_EXEC = openmp_mul
SERIAL_GRAPHMINOR_EXEC = serial_gm
PTHREAD_GRAPHMINOR_EXEC = pthread_gm
OPENMP_GRAPHMINOR_EXEC = openmp_gm


# Rule to build the serial matrix multiplication executable
$(SERIAL_MUL_EXEC): serial_mul.o mmio.o
	$(CC) -o $@ $^

# Rule to build the pthreads matrix multiplication executable
$(PTHREAD_MUL_EXEC): pthread_mul.o mmio.o
	$(CC) -pthread -o $@ $^

# Rule to build the openmp matrix multiplication executable
$(OPENMP_MUL_EXEC): openmp_mul.o mmio.o
	$(CC) -fopenmp -o $@ $^

# Rule to build the serial graphminor executable
$(SERIAL_GRAPHMINOR_EXEC): serial_gm.o mmio.o
	$(CC) -o $@ $^

# Rule to build the pthreads graphminor executable
$(PTHREAD_GRAPHMINOR_EXEC): pthread_gm.o mmio.o
	$(CC) -pthread -o $@ $^

# Rule to build the openmp graphminor executable
$(OPENMP_GRAPHMINOR_EXEC): openmp_gm.o mmio.o
	$(CC) -fopenmp -o $@ $^

# General rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(SERIAL_EXEC) $(PTHREAD_EXEC) $(OPENMP_EXEC) $(SERIAL_GRAPHMINOR_EXEC) $(PTHREAD_GRAPHMINOR_EXEC) $(OPENMP_GRAPHMINOR_EXEC)
