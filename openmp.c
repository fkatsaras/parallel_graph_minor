#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include <omp.h>
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

int readSparseMatrix(const char *filename, SparseMatrixCOO *mat) {
    int ret = mm_read_unsymmetric_sparse(filename, &mat->M, &mat->N, &mat->nnz, &mat->val, &mat->I, &mat->J);
    if (ret!=0) {
        fprintf(stderr, "Failed to read the matrix from file%s\n", filename);
        return ret;
    }
    return 0;
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

// Function to free the memory allocated for a Sparse Matrix
void freeSparseMatrix(SparseMatrixCOO *mat) {
    free(mat->I);
    free(mat->J);
    free(mat->val);
}

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

    // Use OpenMP to parallelize the outer loop
    #pragma omp parallel
    {
        int *local_I = (int *)malloc(capacity * sizeof(int));
        int *local_J = (int *)malloc(capacity * sizeof(int));
        double *local_val = (double *)malloc(capacity * sizeof(double));
        int local_nnz = 0;
        int local_capacity = capacity;

        #pragma omp for nowait
        for (int i = 0; i < A->nnz; i++) {
            for (int j = 0; j < B->nnz; j++) {
                if (A->J[i] == B->I[j]) {
                    int row = A->I[i];
                    int col = B->J[j];
                    double value = A->val[i] * B->val[j];

                    // Check if the entry already exists in the local result
                    bool found = false;
                    for (int k = 0; k < local_nnz; k++) {
                        if (local_I[k] == row && local_J[k] == col) {
                            local_val[k] += value;
                            found = true;
                            break;
                        }
                    }

                    // If it doesn't exist, add a new entry
                    if (!found) {
                        if (local_nnz == local_capacity) {
                            local_capacity *= 2;
                            local_I = (int *)realloc(local_I, local_capacity * sizeof(int));
                            local_J = (int *)realloc(local_J, local_capacity * sizeof(int));
                            local_val = (double *)realloc(local_val, local_capacity * sizeof(double));
                        }
                        local_I[local_nnz] = row;
                        local_J[local_nnz] = col;
                        local_val[local_nnz] = value;
                        local_nnz++;
                    }
                }
            }
        }

        // Use a critical section to merge local results into the global result
        #pragma omp critical
        {
            for (int k = 0; k < local_nnz; k++) {
                bool found = false;
                for (int l = 0; l < C.nnz; l++) {
                    if (C.I[l] == local_I[k] && C.J[l] == local_J[k]) {
                        C.val[l] += local_val[k];
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    if (C.nnz == capacity) {
                        capacity *= 2;
                        C.I = (int *)realloc(C.I, capacity * sizeof(int));
                        C.J = (int *)realloc(C.J, capacity * sizeof(int));
                        C.val = (double *)realloc(C.val, capacity * sizeof(double));
                    }
                    C.I[C.nnz] = local_I[k];
                    C.J[C.nnz] = local_J[k];
                    C.val[C.nnz] = local_val[k];
                    C.nnz++;
                }
            }
        }

        free(local_I);
        free(local_J);
        free(local_val);
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