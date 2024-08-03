#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
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

    int nnz = 0;
    for (int i = 0; i < A->nnz; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {

                C.I[nnz] = A->I[i];
                C.J[nnz] = B->J[j];
                C.val[nnz] = A->val[i] * B->val[j];
                nnz++;
            }
        }
    }

    C.nnz = nnz; // Adjust nnz to the actual number of non zero elements
    C.I = (int *)realloc(C.I, nnz * sizeof(int));
    C.J = (int *)realloc(C.J, nnz * sizeof(int));
    C.val = (double *)realloc(C.val, nnz * sizeof(double));

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
    if (readSparseMatrix("A.mtx", &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix("B.mtx", &B) != 0) {
        return EXIT_FAILURE;
    }

    // Multiply A and B
    C = multiplySparseMatrix(&A, &B);

    // Print the result
    printSparseMatrix(&C, true);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}