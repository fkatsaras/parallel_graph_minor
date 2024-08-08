#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCOO *A, SparseMatrixCOO *B) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int initialCapacity = A->nnz;
    HashTable globalTable;
    initHashTable(&globalTable, initialCapacity);

    #pragma omp parallel
    {
        HashTable localTable;
        initHashTable(&localTable, initialCapacity);

        #pragma omp for 
        for (int i = 0; i < A->nnz; i++) {
            for (int j = 0; j < B->nnz; j++) {
                if (A->J[i] == B->I[j]) {
                    int row = A->I[i];
                    int col = B->J[j];
                    double value = A->val[i] * B->val[j];
                    hashTableInsert(&localTable, row, col, value);
                }
            }
        }

        #pragma omp critical
        {
            mergeHashTables(&globalTable, &localTable);  // Merge local table into global table
        }
        free(localTable.entries);  // Free local hash table entries
    }

    SparseMatrixCOO C = hashTableToSparseMatrix(&globalTable, A->M, B->N);
    printf("I> Hash Table collision count: %d\n", globalTable.collisionCount);
    free(globalTable.entries);  // Free global hash table entries

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
    C = multiplySparseMatrix(&A, &B);

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