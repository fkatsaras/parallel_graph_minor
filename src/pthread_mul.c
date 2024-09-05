#include <stdio.h>

#include "util.c"

#define MAX_THREADS 4

void* threadMultiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    SparseMatrixCSR *A_csr = data->A_csr;
    SparseMatrixCSR *B_csr = data->B_csr;
    HashTable *table = data->table;

    // Perform multiplication CSR
    for (int i = data->start; i < data->end; i++) { // Iterate over rows of A that are assigned to thread %d
        for (int j = A_csr->I_ptr[i]; j < A_csr->I_ptr[i + 1]; j++) { // Iterate over non-zeros in row i of A
            int aCol = A_csr->J[j]; // Column index in A
            double aVal = A_csr->val[j]; // Value in A

            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) { // Iterate over non-zeros in column aCol of B
                int bRow = B_csr->I_ptr[k]; // Row index in B
                int bCol = B_csr->J[k]; // Column index in B
                double bVal = B_csr->val[k]; // Value in B
                double cVal = aVal * bVal;

                // Insert into hash table
                hashTableInsert_L(data, table, i, bCol, cVal);
            }
        }
    }

    return NULL;
}

SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCSR *A_csr, SparseMatrixCSR *B_csr, int numThreads) {
    if (A_csr->N != B_csr->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }
 
    HashTable *table = createHashTable_L(A_csr->nz + B_csr->nz);

    // Initialize threads and calculate workload
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int chunkSize = (A_csr->M + numThreads - 1) / numThreads;   // Split the rows of A to distribute


    for (int i = 0; i < numThreads; i++) {
        threadData[i].thread_id = i;
        threadData[i].A_csr = A_csr;
        threadData[i].B_csr = B_csr;
        threadData[i].table = table;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i + 1) * chunkSize;
        threadData[i].index = 0;        // Empty value for index; Will be initialized later
        if (threadData[i].end > A_csr->M) {
            threadData[i].end = A_csr->M;
        }

        printf("<I> Thread %d processing rows from %d to %d\n", threadData[i].thread_id, threadData[i].start, threadData[i].end);
        // Create thread and assign work
        pthread_create(&threads[i], NULL, threadMultiply, &threadData[i]);
    }

    // Join threads after work has been done
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    Timer DOCtoCOOtime;
    startTimer(&DOCtoCOOtime);

    SparseMatrixCOO C = hashTableToSparseMatrix(table, A_csr->M, B_csr->N);

    stopTimer(&DOCtoCOOtime);

    printf("<I> Hash Table collision count: %d\n", table->collisionCount);
    printElapsedTime(&DOCtoCOOtime, "<I> DOK to COO conversion");

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
    Timer totalTimer;

    // Read from files
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix(matrix_file_B, &B) != 0) {
        freeSparseMatrix(&A); // Free A if B read fails
        return EXIT_FAILURE;
    }

    SparseMatrixCSR A_csr, B_csr;
    // Convert COO matrices to CSR format
    A_csr = COOtoCSR(A);
    B_csr = COOtoCSR(B);
    startTimer(&totalTimer);

    // Multiply A and B csr and return C coo
    C = multiplySparseMatrixParallel(&A_csr, &B_csr, 4);

    stopTimer(&totalTimer);
    // Print the result
    printSparseMatrix("C", &C, true);

    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&C);
    }

    printElapsedTime(&totalTimer,"Total multiplication");

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}