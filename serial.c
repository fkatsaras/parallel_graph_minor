#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCSR *A, SparseMatrixCSR *B) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int initialCapacity = A->nnz; // Initial estimation for size (Greater initial est -> Less hash table resizes -> Greater chance of getting segfault)
    HashTable table;
    initHashTable(&table, initialCapacity);

    for (int i = 0; i < A->M; i++) {
        for (int aj = A->I_ptr[i]; aj < A->I_ptr[i + 1]; aj++) {
            int A_col = A->J[aj];
            for (int bj = B->I_ptr[A_col]; bj < B->I_ptr[A_col + 1]; bj++) {
                int row = i;
                int col = B->J[bj];
                double value = A->val[aj] * B->val[bj];
                hashTableInsert(&table, row, col, value);
            }
        }
    }

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

    SparseMatrixCSR D = COOtoCSR(A);
    SparseMatrixCSR E = COOtoCSR(B);

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // Multiply A and B
    C = multiplySparseMatrix(&D, &E);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Print the result
    printSparseMatrix(&C, true);

    printf("\nI> Total multiplication execution time: %f seconds\n", cpu_time_used);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    freeSparseMatrixCSR(&D);
    freeSparseMatrixCSR(&E);

    return 0;
}