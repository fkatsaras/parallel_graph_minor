#include <stdio.h>

#include "util.c"

#define MAX_THREADS 4

void* threadMultiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    SparseMatrixCSR *A_csr = data->A_csr;
    SparseMatrixCSR *B_csr = data->B_csr;
    HashTable *table = data->table;

    // Perform multiplication CSR
    // Iterate over the assigned submatrix of A
    for (int i = data->row_start; i < data->row_end; i++) { // Iterate over rows of A that are assigned to thread
        for (int j = A_csr->I_ptr[i]; j < A_csr->I_ptr[i + 1]; j++) { // Iterate over non-zeros in row i of A
            int aCol = A_csr->J[j]; // Column index in A
            double aVal = A_csr->val[j]; // Value in A

            // Iterate over the assigned submatrix of B
            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) { // Iterate over non-zeros in column aCol of B
                int bCol = B_csr->J[k]; // Column index in B
                double bVal = B_csr->val[k]; // Value in B
                double cVal = aVal * bVal;

                // Insert into hash table
                hashTableInsert(table, i, bCol, cVal, false);
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

    // Initialize threads and calculate workload
    pthread_t threads[numThreads];
    ThreadData threadDataCompute[numThreads];
    int row_block_size = (A_csr->M + numThreads - 1) / numThreads;

    // Create private hash tables for each thread
    HashTable *tables[numThreads];

    for (int i = 0; i < numThreads; i++) {
        tables[i] = createHashTable(4 * A_csr->nz); // Each thread gets its private hash table
        threadDataCompute[i].thread_id = i;
        threadDataCompute[i].A_csr = A_csr;
        threadDataCompute[i].B_csr = B_csr;
        threadDataCompute[i].table = tables[i];

        // Assign row ranges for threads
        threadDataCompute[i].row_start = i * row_block_size;
        threadDataCompute[i].row_end = (i + 1) * row_block_size;
        if (threadDataCompute[i].row_end > A_csr->M) threadDataCompute[i].row_end = A_csr->M;

        printf("Thread %d started processing rows from %d to %d\n", i, threadDataCompute[i].row_start, threadDataCompute[i].row_end);

        // Create thread
        pthread_create(&threads[i], NULL, threadMultiply, &threadDataCompute[i]);
    }

    // Join threads after work has been done
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    Timer DOCtoCOOtime;
    startTimer(&DOCtoCOOtime);

    SparseMatrixCOO C = hashTablesToSparseMatrix(tables, numThreads, A_csr->M, B_csr->N);

    stopTimer(&DOCtoCOOtime);

    // printf("<I> Hash Table collision count: %d\n", table->collisionCount);
    printElapsedTime(&DOCtoCOOtime, "<I> DOK to COO conversion");

    // Free allocated memory
    for (int i = 0; i < numThreads; i++) {
        freeHashTable(tables[i]);
    }
    freeCSRMatrix(A_csr);
    freeCSRMatrix(B_csr);

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
    if (argc == 4 && strcmp(argv[3], "--pprint") == 0) {
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

    SparseMatrixCSR A_csr, B_csr;
    // Convert COO matrices to CSR format
    A_csr = COOtoCSR(A);
    B_csr = COOtoCSR(B);

    Timer totalTimer;
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

    freeSparseMatrix(&C);

    return 0;
}