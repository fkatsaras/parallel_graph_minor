#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include "mmio.h"

#define MAX_THREADS 2

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

pthread_mutex_t mutex; // Global mutex

// Struct to hold thread information 
typedef struct {
    int thread_id;
    SparseMatrixCOO *A;
    SparseMatrixCOO *B;
    SparseMatrixCOO *C;
    int start;
    int end;
    pthread_mutex_t *mutex;
} ThreadData;

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

void* multiplyPart(void *arg) {
    ThreadData *data = (ThreadData*) arg;
    SparseMatrixCOO *A = data->A;
    SparseMatrixCOO *B = data->B;
    SparseMatrixCOO *C = data->C;
    int start = data->start;
    int end = data->end;

    printf("Thread %d processing elements", data->thread_id);
    for (int i = data->start; i < data->end; i++) {
        printf("\n( %d , %d ) = %f", data->A->I[i], data->A->J[i], data->A->val[i]);
    }

    for (int i = start; i < end; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {

                printf("\nFound matching elemets: A(%d , %d) * B(%d, %d) -> %f * %f",A->I[i], A->J[i], B->I[j], B->J[j], A->val[i], B->val[j]);
                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];
                printf("\tMultiplication result: %f", value);
                printf("\nC nnz before new element: %d\n\n", C->nnz);

                // Lock before modifying shared data
                pthread_mutex_lock(data->mutex);

                // Check if the entry already exists in the result
                int k;
                for (k = 0; k < C->nnz; k++) {
                    if (C->I[k] == row && C->J[k] == col) {

                        printf("Entry already exists ( %d , %d ) = %f : Adding %f to sum\n", row, col, C->val[k], value);
                        C->val[k] += value;
                        break;
                    }
                }

                // If it doesn't exist, add a new entry
                if (k == C->nnz) {

                    printf("New entry added: ( %d, %d ) = %f\n", row, col, value);
                    C->I[C->nnz] = row;
                    C->J[C->nnz] = col;
                    C->val[C->nnz] = value;
                    C->nnz++;
                }

                // Unlock after modifying shared data
                pthread_mutex_unlock(data->mutex);
            }
        }
    }
    printf("Thread %d finished processing\n", data->thread_id);
    pthread_exit(NULL);
}

// Function to multiply two sparse matrices in COO format using pthreads
SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCOO *A, SparseMatrixCOO *B) {
    // Step 1 : Initialization
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication. \n");
        exit(EXIT_FAILURE);
    }

    SparseMatrixCOO C;
    initSparseMatrix(&C, A->M, B->N, A->nnz * B->nnz);

    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    int elements_per_thread = (A->nnz + MAX_THREADS - 1) / MAX_THREADS; // Divide work among threads
    // Declare mutex
    pthread_mutex_t mutex;

    pthread_mutex_init(&mutex, NULL);

    // Initialize thread data and create threads
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = &C;
        thread_data[i].start = i * elements_per_thread;
        thread_data[i].end = (i + 1) * elements_per_thread;
        if (thread_data[i].end > A->nnz) thread_data[i].end = A->nnz;
        thread_data[i].mutex = &mutex;

        printf("Creating thread %d to process elements from %d to %d\n", i, thread_data[i].start, thread_data[i].end);
        if (pthread_create(&threads[i], NULL, multiplyPart, (void *)&thread_data[i])) {
            printf("Error creating thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Join threads after work has been done
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
        printf("Thread %d has finished\n", i);
    }

    pthread_mutex_destroy(&mutex);

    printf("Non-zero elements in C before realloc: %d\n", C.nnz);
    for (int i = 0; i < C.nnz; i++) {
        printf("C.I[%d] = %d, C.J[%d] = %d, C.val[%d] = %f\n", i, C.I[i], i, C.J[i], i, C.val[i]);
    }

    C.I = (int *)realloc(C.I, C.nnz * sizeof(int));
    C.J = (int *)realloc(C.J, C.nnz * sizeof(int));
    C.val = (double *)realloc(C.val, C.nnz * sizeof(double));

    printf("Multiplication complete with %d non-zero elements in the result\n", C.nnz);

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

    int numThreads = 2;

    // EXAMPLE USAGE
    if (readSparseMatrix("A.mtx", &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix("B.mtx", &B) != 0) {
        return EXIT_FAILURE;
    }

    // Multiply A and B
    C = multiplySparseMatrixParallel(&A, &B);

    // Print the result
    printSparseMatrix(&C, 1);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}