#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>


typedef struct {
    int row, col;
} HashKey;

typedef struct {
    HashKey key;
    double value;
    bool occupied;
} HashEntry;

typedef struct {
    HashEntry *entries;
    int capacity;
    int size;
    int collisionCount;
} HashTable;

typedef struct SparseMatrixCSR{

    int *I_ptr;
    int *J;
    long double *val;

    int M;
    int N;
    int nz;
} SparseMatrixCSR;

typedef struct
{
    /*         
            N , J
    +-------------------+
    | *                 |
    |    *    *         |
    |    *  *           | M, I
    |          *  *   * |
    |             *     |
    |      *   *     *  |
    +-------------------+
    
    nnz = *
    (I[nnz], J[nnz]) = value
    
    Legend:
    M = Rows
    N = Columns
    I = Row indices
    J = Column indices
   */
    int M, N, nnz; // Matrix size info

    int *I, *J; // Matrix value info 
    double *val
} SparseMatrixCOO;

// Struct to hold thread information 
typedef struct {
    int thread_id;
    SparseMatrixCOO *A;
    SparseMatrixCOO *B;
    SparseMatrixCOO *C;
    int start;
    int end;
    pthread_mutex_t *mutex;
    HashTable *table;
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

// Function to print a sparse matrix in COO format
void printSparseMatrix(SparseMatrixCOO *mat, bool head) {
    printf(" SparseMatrixCOO(shape = ( %d , %d ), nnz = %d )\n", mat->M, mat->N, mat->nnz );
    if (head) {
        if (mat->nnz > 10) {
            for (int i = 0; i < 10; i++) {
            printf("\t( %d , %d ) = %f\n", mat->I[i], mat->J[i], mat->val[i] );
            }
        }
        else {
            for (int i = 0; i < mat->nnz; i++) {
            printf("\t( %d , %d ) = %f\n", mat->I[i], mat->J[i], mat->val[i] );
            }
        }   
    }
}

// Function to print a sparse matrix in CSR format
void printSparseMatrixCSR(SparseMatrixCSR *mat, bool head) {
    printf("SparseMatrixCSR(shape = ( %d , %d ), nz = %d )\n", mat->M, mat->N, mat->nz);
    int count = 0;
    for (int i = 0; i < mat->M; i++) {
        for (int j = mat->I_ptr[i]; j < mat->I_ptr[i + 1]; j++) {
            if (head && count >= 10) {
                return;
            }
            printf("\t( %d , %d ) = %f\n", i, mat->J[j], mat->val[j]);
            count++;
        }
    }
}

int readSparseMatrix(const char *filename, SparseMatrixCOO *mat) {
    int ret = mm_read_unsymmetric_sparse(filename, &mat->M, &mat->N, &mat->nnz, &mat->val, &mat->I, &mat->J);
    if (ret!=0) {
        fprintf(stderr, "\tFailed to read the matrix from file %s\n", filename);
        return ret;
    }
    return 0;
}

// Converts COO to CSR
SparseMatrixCSR COOtoCSR(SparseMatrixCOO  A){

    SparseMatrixCSR output;
    output.nz = A.nnz;
    output.M = A.M;
    output.N = A.N;

    output.I_ptr = (int *) malloc((output.M + 1) * sizeof(int));
    output.J = (int *) malloc(output.nz * sizeof(int));
    output.val = (long double *) malloc(output.nz * sizeof(long double));

    for (int i = 0; i < (output.M + 1); i++){
        output.I_ptr[i] = 0;
    }

    for (int i = 0; i < A.nnz; i++){
        output.val[i] = A.val[i];
        output.J[i] = A.J[i];
        output.I_ptr[A.I[i] + 1]++;
    }

    for (int i = 0; i < output.M; i++){
        output.I_ptr[i + 1] += output.I_ptr[i];
    }

    return output;
}

// Default hash function for HashKey
unsigned int hash(HashKey key, int capacity) {
    unsigned int hashValue = key.row * 31 + key.col;
    return hashValue % capacity;
}

// FNV1 hash function for HashKey
unsigned int fnv1aHash(int row, int col, int capacity) {
    unsigned int hash = 2166136261u;
    hash ^= row;
    hash *= 16777619u;
    hash ^= col;
    hash *= 16777619u;
    return hash % capacity;
}

// DJB2 hash function for HashKey
unsigned int djb2Hash(int row, int col, int capacity) {
    unsigned int hash = 5381;
    hash = ((hash << 5) + hash) ^ row;
    hash = ((hash << 5) + hash) ^ col;
    return hash % capacity;
}

// Initialize the hash table
void initHashTable(HashTable *table, int capacity) {
    table->capacity = capacity;
    table->size = 0;
    table->entries = (HashEntry *)calloc(capacity, sizeof(HashEntry));
    table->collisionCount = 0;
}

void resizeHashTable(HashTable *table);

// Insert or update an entry in the hash table
void hashTableInsert(HashTable *table, int row, int col, double value) {
    if (table->size > table->capacity * 0.7) {
        printf("Resized Hash table\n");
        resizeHashTable(table);
    }

    HashKey key = { row, col };
    // unsigned int index = hash(key, table->capacity);
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);

    // Linear probing for collision resolution
    while (table->entries[index].occupied) {
        table->collisionCount++;
        if (table->entries[index].key.row == row && table->entries[index].key.col == col) {
            table->entries[index].value += value;
            return;
        }
        index = (index + 1) % table->capacity;
    }

    table->entries[index].key = key;
    table->entries[index].value = value;
    table->entries[index].occupied = true;
    table->size++;
}

void resizeHashTable(HashTable *table) {
    int oldCapacity = table->capacity;
    HashEntry *oldEntries = table->entries;

    table->capacity *= 2;
    table->entries = (HashEntry *)calloc(table->capacity, sizeof(HashEntry));
    table->size = 0;

    for (int i = 0; i < oldCapacity; i++) {
        if (oldEntries[i].occupied) {
            hashTableInsert(table, oldEntries[i].key.row, oldEntries[i].key.col, oldEntries[i].value);
        }
    }

    free(oldEntries);
}

// Convert hash table to sparse matrix COO format
SparseMatrixCOO hashTableToSparseMatrix(HashTable *table, int M, int N) {
    SparseMatrixCOO C;
    initSparseMatrix(&C, M, N, table->size);
    int index = 0;
    for (int i = 0; i < table->capacity; i++) {
        if (table->entries[i].key.row != 0 || table->entries[i].key.col != 0) {
            C.I[index] = table->entries[i].key.row;
            C.J[index] = table->entries[i].key.col;
            C.val[index] = table->entries[i].value;
            index++;
        }
    }
    C.nnz = index;
    return C;
}