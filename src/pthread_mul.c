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
    for (int i = data->row_start; i < data->row_end; i++) { // Iterate over rows of A that are assigned to thread %d
        for (int j = A_csr->I_ptr[i]; j < A_csr->I_ptr[i + 1]; j++) { // Iterate over non-zeros in row i of A
            int aCol = A_csr->J[j]; // Column index in A
            double aVal = A_csr->val[j]; // Value in A

            // Iterate over the assigned submatrix of B
            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) { // Iterate over non-zeros in column aCol of B
                int bCol = B_csr->J[k]; // Column index in B

                if (bCol >= data->col_start && bCol < data->col_end) { // Check if within the assigned column block
                    double bVal = B_csr->val[k]; // Value in B
                    double cVal = aVal * bVal;

                    // Insert into hash table
                    hashTableInsert(table, i, bCol, cVal, false);
                }
            }
        }
    }

    return NULL;
}

void* threadMerge(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    HashTable *table = data->globalTable;

    mergeHashTables_L(data, table, false);

    return NULL;
}

SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCSR *A_csr, SparseMatrixCSR *B_csr, int numThreads) {
    if (A_csr->N != B_csr->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int row_blocks = 2; // Number of row blocks for A
    int col_blocks = 2; // Number of column blocks for B
    int total_blocks = row_blocks * col_blocks;

    // Initialize threads and calculate workload
    pthread_t threads[total_blocks];
    ThreadData threadDataCompute[total_blocks];
    // int chunkSize = (A_csr->M + numThreads - 1) / numThreads;   // Split the rows of A to distribute
    int row_block_size = (A_csr->M + row_blocks - 1)/ row_blocks;
    int col_block_size = (B_csr->N + col_blocks - 1)/ col_blocks;

    // Create private hash tables for each thread
    HashTable *tables[numThreads];

    // Create global hash table for final result
    // HashTable *table = createHashTable_L(3 * A_csr->nz);
    HashTable *table = createHashTable(3 * A_csr->nz);

    for (int i = 0; i < row_blocks; i++) {
        for (int j = 0; j < col_blocks; j++) {
            int thread_id = i * col_blocks + j;
            tables[thread_id] = createHashTable(4 * A_csr->nz); // Each thread gets its private hash table
            threadDataCompute[thread_id].thread_id = thread_id;
            threadDataCompute[thread_id].A_csr = A_csr;
            threadDataCompute[thread_id].B_csr = B_csr;
            threadDataCompute[thread_id].table = tables[thread_id];
            threadDataCompute[thread_id].globalTable = table;
            
            // Assign ranges of submatrices
            threadDataCompute[thread_id].row_start = i * row_block_size;
            threadDataCompute[thread_id].row_end = (i + 1) * row_block_size;
            if (threadDataCompute[thread_id].row_end > A_csr->M) threadDataCompute[thread_id].row_end = A_csr->M;
            
            threadDataCompute[thread_id].col_start = j * col_block_size;
            threadDataCompute[thread_id].col_end = (j + 1) * col_block_size;
            if (threadDataCompute[thread_id].col_end > B_csr->N) threadDataCompute[thread_id].col_end = B_csr->N;

            printf("Thread %d started processing rows from %d ... %d, cols from %d ... %d\n",thread_id, threadDataCompute[thread_id].row_start, threadDataCompute[thread_id].row_end, threadDataCompute[thread_id].col_start, threadDataCompute[thread_id].col_end);
    
            // Create thread
            pthread_create(&threads[thread_id], NULL, threadMultiply, &threadDataCompute[thread_id]);
        }
    }

    // Join threads after work has been done
    for (int i = 0; i < total_blocks; i++) {
        pthread_join(threads[i], NULL);
    }

    mergeHashTables(tables, table, numThreads, false);

    Timer DOCtoCOOtime;
    startTimer(&DOCtoCOOtime);

    SparseMatrixCOO C = hashTableToSparseMatrix(table, A_csr->M, B_csr->N);

    stopTimer(&DOCtoCOOtime);

    printf("<I> Hash Table collision count: %d\n", table->collisionCount);
    printElapsedTime(&DOCtoCOOtime, "<I> DOK to COO conversion");

    // // Free allocated memory
    // for (int i = 0; i < numThreads; i++) {
    //     freeHashTable(tables[i]);
    // }
    freeCSRMatrix(A_csr);
    freeCSRMatrix(B_csr);
    freeHashTable(table); 

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

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}