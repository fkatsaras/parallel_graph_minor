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

    for (int i = start; i < end; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {

                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];

                // Lock before modifying shared data
                pthread_mutex_lock(data->mutex);

                // Check if the entry already exists in the result
                int k;
                for (k = 0; k < C->nnz; k++) {
                    if (C->I[k] == row && C->J[k] == col) {

                        C->val[k] += value;
                        break;
                    }
                }

                // If it doesn't exist, add a new entry
                if (k == C->nnz) {

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

    C.nnz = 0;

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

    C.I = (int *)realloc(C.I, C.nnz * sizeof(int));
    C.J = (int *)realloc(C.J, C.nnz * sizeof(int));
    C.val = (double *)realloc(C.val, C.nnz * sizeof(double));

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
    if (readSparseMatrix("GD97_b/GD97_b.mtx", &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix("GD97_b/GD97_b.mtx", &B) != 0) {
        return EXIT_FAILURE;
    }

    // Multiply A and B
    C = multiplySparseMatrixParallel(&A, &B);

    // Print the result
    printSparseMatrix(&C, 0);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}