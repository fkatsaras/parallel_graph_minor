#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "mmio.h"


typedef struct SparseMatrixCOO
{
    /*         
            M , I
    +-------------------+
    | *                 |
    |    *    *         |
    |    *  *           | N, J
    |          *  *   * |
    |             *     |
    |      *   *     *  |
    +-------------------+
    
    nnz = *
    (I[nnz], J[nnz]) = value
    
    Legend:
    N = Rows
    M = Columns
    I = Row indices
    J = Column indices
   */
    int M, N, nnz; // Matrix size info

    int *I, *J; // Matrix value info 
    double *val
} SparseMatrixCOO;

// Function to initialize a sparse matrix
void initSparseMatrix(SparseMatrixCOO *mat, int M, int N, int nnz) {

    mat->M = M;
    mat->N = N;
    mat->nnz = nnz;
    
    mat->I = (int *)malloc(nnz * sizeof(int));
    mat->J = (int *)malloc(nnz * sizeof(int));
    mat->val = (double *)malloc(nnz * sizeof(double));
}

// Function to free the memory allocated for a Sparse Matrix
void freeSparseMatrix(SparseMatrixCOO *mat) {
    free(mat->I);
    free(mat->J);
    free(mat->val);
}

// Function to multiply two sparse matrices in COO format
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCOO *A, SparseMatrixCOO *B) {
    // Step 1 : Initialization
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication. \n");
        exit(EXIT_FAILURE);
    }

    // Step 2 : Multiplication process (Upper bound for non-zero elements nnzA * nnzB)
    SparseMatrixCOO C;
    initSparseMatrix(&C, A->M, B->N, A->nnz * B->nnz);

    // Step 2.1: Using a temporary structure to accumulate values
    int *temp_I = (int *)malloc(A->nnz * B->nnz * sizeof(int));
    int *temp_J = (int *)malloc(A->nnz * B->nnz * sizeof(int));
    double *temp_val = (double *)calloc(A->nnz * B->nnz, sizeof(double)); // initialize to 0

    int count = 0;
    for (int i = 0; i < A->nnz; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {
                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];

                // Check if the entry already exists in the result
                int k;
                for (k = 0; k < count; k++) {
                    if (temp_I[k] == row && temp_J[k] == col) {
                        temp_val[k] += value;
                        break;
                    }
                }

                // If it doesn't exist, add a new entry
                if (k == count) {
                    temp_I[count] = row;
                    temp_J[count] = col;
                    temp_val[count] = value;
                    count++;
                }
            }
        }
    }

    // Step 3: Assign values to result matrix C
    C.nnz = count;
    C.I = (int *)realloc(C.I, count * sizeof(int));
    C.J = (int *)realloc(C.J, count * sizeof(int));
    C.val = (double *)realloc(C.val, count * sizeof(double));

    for (int i = 0; i < count; i++) {
        C.I[i] = temp_I[i];
        C.J[i] = temp_J[i];
        C.val[i] = temp_val[i];
    }

    free(temp_I);
    free(temp_J);
    free(temp_val);

    return C;
}

// Function to print a sparse matrix in COO format
void printSparseMatrix(SparseMatrixCOO *mat, bool full) {
    printf(" SparseMatrixCOO(shape = ( %d , %d ), nnz = %d )\n", mat->M, mat->N, mat->nnz );
    if (full) {
        for (int i = 0; i < mat->nnz; i++) {
            printf("\t( %d , %d ) = %f\n", mat->I[i], mat->J[i], mat->val[i] );
        }
    }
}

int readSparseMatrix(const char *filename, SparseMatrixCOO *mat) {
    int ret = mm_read_unsymmetric_sparse(filename, &mat->M, &mat->N, &mat->nnz, &mat->val, &mat->I, &mat->J);
    if (ret!=0) {
        fprintf(stderr, "Failed to read the matrix from file%s\n", filename);
        return ret;
    }
    return 0;
}

int main() {

    SparseMatrixCOO A, B, C;

    // EXAMPLE USAGE
    if (readSparseMatrix("GD97_b/GD97_b.mtx", &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix("GD97_b/GD97_b.mtx", &B) != 0) {
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