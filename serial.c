#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mmio.h"
#include "util.c"


// Function to multiply two sparse matrices in COO format
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCOO *A, SparseMatrixCOO *B) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    SparseMatrixCOO C;
    initSparseMatrix(&C, A->M, B->N, 0);

    int capacity = 1024; 
    C.I = (int *)malloc(capacity * sizeof(int));
    C.J = (int *)malloc(capacity * sizeof(int));
    C.val = (double *)malloc(capacity * sizeof(double));

    for (int i = 0; i < A->nnz; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {
                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];

                // Check if the entry already exists in the result
                bool found = false;
                for (int k = 0; k < C.nnz; k++) {
                    if (C.I[k] == row && C.J[k] == col) {
                        C.val[k] += value;
                        found = true;
                        break;
                    }
                }

                // If it doesn't exist, add a new entry
                if (!found) {
                    if (C.nnz == capacity) {
                        // Dynamically allocate new memory for the new entries
                        capacity *= 2;
                        C.I = (int *)realloc(C.I, capacity * sizeof(int));
                        C.J = (int *)realloc(C.J, capacity * sizeof(int));
                        C.val = (double *)realloc(C.val, capacity * sizeof(double));
                    }
                    C.I[C.nnz] = row;
                    C.J[C.nnz] = col;
                    C.val[C.nnz] = value;
                    C.nnz++;
                }
            }
        }
    }

    // Shrink the arrays to the actual size needed
    C.I = (int *)realloc(C.I, C.nnz * sizeof(int));
    C.J = (int *)realloc(C.J, C.nnz * sizeof(int));
    C.val = (double *)realloc(C.val, C.nnz * sizeof(double));

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
    
    if (readSparseMatrix(matrix_file_A, &B) != 0) {
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
    printf("Execution time: %f seconds\n", cpu_time_used);

    // Print the result
    printSparseMatrix(&C, false);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}