#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCOO *A, SparseMatrixCOO *B) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int initialCapacity = A->nnz; // Initial estimation for size (Greater initial est -> Less hash table resizes -> Greater chance of getting segfault)
    HashTable table;
    initHashTable(&table, initialCapacity);

    #pragma omp parallel for shared(A, B, table) private(i, j, row, col, value)
    for (int i = 0; i < A->nnz; i++) {
        #pragma omp for
        {
            for (int j = 0; j < B->nnz; j++) {
                if (A->J[i] == B->I[j]) {
                    int row = A->I[i];
                    int col = B->J[j];
                    double value = A->val[i] * B->val[j];
                    #pragma omp critical
                    {
                        hashTableInsert(&table, row, col, value);
                    }
                }
            }
        }
    }

    SparseMatrixCOO C = hashTableToSparseMatrix(&table, A->M, B->N);
    printf("I> Hash Table collision count: %d\n", table.collisionCount);
    free(table.entries);  // Free the hash table entries

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
    C = multiplySparseMatrixParallel(&A, &B);

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
