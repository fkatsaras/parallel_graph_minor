#include <stdio.h>

#include "util.c"

#define MAX_THREADS 4

void* threadMultiply(void* arg) {
    ThreadCSRData* data = (ThreadCSRData*)arg;
    SparseMatrixCSR *A = data->A;
    SparseMatrixCSR *B = data->B;
    HashTable *table = data->table;

    for (int i = data->start; i < data->end; i++) {
        for (int aj = A->I_ptr[i]; aj < A->I_ptr[i + 1]; aj++) {
            int A_col = A->J[aj];
            for (int bj = B->I_ptr[A_col]; bj < B->I_ptr[A_col + 1]; bj++) {
                int row = i;
                int col = B->J[bj];
                double value = A->val[aj] * B->val[bj];

                pthread_mutex_lock(data->mutex);  // Lock before modifying shared data

                hashTableInsert(table, row, col, value);

                pthread_mutex_unlock(data->mutex); // Unlock after modifying shared data
            }
        }
    }

    return NULL;
}

SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCSR *A, SparseMatrixCSR *B, int numThreads) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int initialCapacity = A->nnz; 
    HashTable table;
    initHashTable(&table, initialCapacity);

    // Initialize threads and calculate workload
    pthread_t threads[numThreads];
    ThreadCSRData threadData[numThreads];
    int chunkSize = (A->M + numThreads - 1) / numThreads;

    // Initialize mutex
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < numThreads; i++) {
        threadData[i].A = A;
        threadData[i].B = B;
        threadData[i].table = &table;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i + 1) * chunkSize;
        if (threadData[i].end > A->M) {
            threadData[i].end = A->M;
        }
        // Assign the mutex to each thread
        threadData[i].mutex = &mutex;
        // Create thread and assign work
        pthread_create(&threads[i], NULL, threadMultiply, &threadData[i]);
    }

    // Join threads after work has been done
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    clock_t hashStart, hashEnd;
    double hashToCOOTime;
    hashStart = clock();

    SparseMatrixCOO C = hashTableToSparseMatrix(&table, A->M, B->N);

    hashEnd = clock();
    hashToCOOTime = ((double) (hashEnd - hashStart)) / CLOCKS_PER_SEC;

    printf("I> Hash Table collision count: %d\n", table.collisionCount);
    printf("I> DOK to COO conversion execution time: %f seconds\n", hashToCOOTime);

    free(table.entries);  // Free the hash table entries

    return C;
}

int main(int argc, char *argv[]) {

    // Input the matrix filename arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file_A> <matrix_file_B>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file_A = argv[1];
    const char *matrix_file_B = argv[2];

    SparseMatrixCOO A, B, C;

    // EXAMPLE USAGE
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix(matrix_file_B, &B) != 0) {
        freeSparseMatrix(&A); // Free A if B read fails
        return EXIT_FAILURE;
    }

    SparseMatrixCSR D, E;
    D = COOtoCSR(A);
    E = COOtoCSR(B);

    clock_t start, end;
    double cpu_time_used, hashToCOOTime;

    start = clock();

    // Multiply A and B
    C = multiplySparseMatrixParallel(&D, &E, MAX_THREADS);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Print the result
    printSparseMatrix(&C, true);

    printf("\nI> Total multiplication execution time: %f seconds\n", cpu_time_used);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}
