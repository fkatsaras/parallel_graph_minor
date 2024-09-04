#include <stdio.h>

#include "util.c"

#define MAX_THREADS 4

void* threadMultiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    SparseMatrixCOO *A = data->A;
    SparseMatrixCOO *B = data->B;
    HashTable *table = data->table;

    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {
                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];

                hashTableInsert_L(data, table, row, col, value);
            }
        }
    }

    return NULL;
}

SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCOO *A, SparseMatrixCOO *B, int numThreads) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }
 
    HashTable *table = createHashTable_L(A->nnz + B->nnz);

    // Initialize threads and calculate workload
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int chunkSize = (A->nnz + numThreads - 1) / numThreads;


    for (int i = 0; i < numThreads; i++) {
        threadData[i].thread_id = i;
        threadData[i].A = A;
        threadData[i].B = B;
        threadData[i].table = table;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i + 1) * chunkSize;
        threadData[i].index = 0;        // Empty value for index; Will be initialized later
        if (threadData[i].end > A->nnz) {
            threadData[i].end = A->nnz;
        }
        // Create thread and assign work
        pthread_create(&threads[i], NULL, threadMultiply, &threadData[i]);
    }

    // Join threads after work has been done
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t hashStart, hashEnd;
    double hashToCOOTime;
    hashStart = clock();

    SparseMatrixCOO C = hashTableToSparseMatrix(table, A->M, B->N);

    hashEnd = clock();
    hashToCOOTime = ((double) (hashEnd - hashStart)) / CLOCKS_PER_SEC;

    printf("I> Hash Table collision count: %d\n", table->collisionCount);
    printf("I> DOK to COO conversion execution time: %f seconds\n", hashToCOOTime);

    freeHashTable(table);  // Free the hash table entries

    return C;
}

int main(int argc, char *argv[]) {

    // Input the matrix filename arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file_A> <matrix_file_B> [-pprint]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file_A = argv[1];
    const char *matrix_file_B = argv[2];
    bool pprint_flag = false;

    // Check for the -pprint flag
    if (argc == 4 && strcmp(argv[3], "-pprint") == 0) {
        pprint_flag = true;
    }

    SparseMatrixCOO A, B, C;

    // Read from files
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix(matrix_file_B, &B) != 0) {
        freeSparseMatrix(&A); // Free A if B read fails
        return EXIT_FAILURE;
    }


    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // Multiply A and B
    C = multiplySparseMatrixParallel(&A, &B, MAX_THREADS);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Print the result
    printSparseMatrix("C", &C, true);

    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&C);
    }

    printf("\nI> Total multiplication execution time: %f seconds\n", cpu_time_used);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}
