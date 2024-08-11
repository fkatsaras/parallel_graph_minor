#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <omp.h>

#include "mmio.h"

/******************************** Hash table structs ***************************** */

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
    pthread_mutex_t *locks; // Array of mutexes for fine-grained locking
    int lockCount;          // Number of locks
} HashTable;

/******************************** Matrix data structs ***************************** */
typedef struct SparseMatrixCSR{

    int *I_ptr;
    int *J;
    double *val;

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

/******************************** Thread data structs ***************************** */

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

unsigned int fnv1aHash(int row, int col, int capacity);

// Function to initialize a sparse matrix
void initSparseMatrix(SparseMatrixCOO *mat, int M, int N, int nnz) {

    mat->M = M;
    mat->N = N;
    mat->nnz = nnz;
    
    mat->I = (int *)malloc(nnz * sizeof(int));
    mat->J = (int *)malloc(nnz * sizeof(int));
    mat->val = (double *)malloc(nnz * sizeof(double));
}

// Function to initialize a CSR matrix
void initSparseMatrixCSR(SparseMatrixCSR *mat, int M, int N, int nz) {
    mat->M = M;
    mat->N = N;
    mat->nz = nz;
    
    mat->I_ptr = (int *)malloc((M + 1) * sizeof(int));
    mat->J = (int *)malloc(nz * sizeof(int));
    mat->val = (double *)malloc(nz * sizeof(double));
}

// Function to free the memory allocated for a Sparse Matrix
void freeSparseMatrix(SparseMatrixCOO *mat) {
    free(mat->I);
    free(mat->J);
    free(mat->val);
}

// Function to print a sparse matrix in COO format
void printSparseMatrix(const char *name, SparseMatrixCOO *mat, bool head) {
    printf(" %s : SparseMatrixCOO(shape = ( %d , %d ), nnz = %d )\n", name, mat->M, mat->N, mat->nnz );
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
    else {
        for (int i = 0; i < mat->nnz; i++) {
            printf("\t( %d , %d ) = %f\n", mat->I[i], mat->J[i], mat->val[i] );
            }
    }
}

// Function to print a sparse matrix in CSR format
void printSparseMatrixCSR(SparseMatrixCSR *mat) {
    printf("CSR Matrix:\n");
    printf("I_ptr: ");
    for (int i = 0; i <= mat->M; i++) {
        printf("%d ", mat->I_ptr[i]);
    }
    printf("\n");

    printf("J: ");
    for (int i = 0; i < mat->nz; i++) {
        printf("%d ", mat->J[i]);
    }
    printf("\n");

    printf("val: ");
    for (int i = 0; i < mat->nz; i++) {
        printf("%f ", mat->val[i]);
    }
    printf("\n");
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
    initSparseMatrixCSR(&output, A.M, A.N, A.nnz);

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

// Secondary hash function for double hashing
unsigned int secondaryHash(int row, int col, int capacity) {
    // Use a prime number less than the table capacity
    return 1 + ((row + col) % (capacity - 1));
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
// MurmurHash3 function for HashKey
uint32_t murmurHash3(uint32_t key1, uint32_t key2, uint32_t capacity) {
    uint32_t seed = 42; // Seed can be any arbitrary value
    uint32_t h1 = seed;
    uint32_t k1 = key1;
    uint32_t k2 = key2;

    // Constants from MurmurHash3
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // Mix key1
    k1 *= c1;
    k1 = (k1 << 15) | (k1 >> (32 - 15)); // ROTL32
    k1 *= c2;

    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >> (32 - 13)); // ROTL32
    h1 = h1 * 5 + 0xe6546b64;

    // Mix key2
    k2 *= c1;
    k2 = (k2 << 15) | (k2 >> (32 - 15)); // ROTL32
    k2 *= c2;

    h1 ^= k2;
    h1 = (h1 << 13) | (h1 >> (32 - 13)); // ROTL32
    h1 = h1 * 5 + 0xe6546b64;

    // Finalization
    h1 ^= 8; // Length of two 32-bit keys (2 * 4 bytes)
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1 % capacity;
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
    if (table->size > table->capacity * 0.7) { // Resize table incase 70% is full
        printf("I> Resized Hash table\n");
        clock_t resizeStart, resizeEnd; 
        resizeStart = clock();
        resizeHashTable(table);
        resizeEnd = clock();
        printf("I> Resize execution time: %f seconds\n", ((double) (resizeEnd - resizeStart)) / CLOCKS_PER_SEC ); 
    }

    HashKey key = { row, col };
    // Primary hash
    // unsigned int index = hash(key, table->capacity);
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);
    // unsigned int index = murmurHash3((uint32_t) key.row, (uint32_t) key.col, (uint32_t) table->capacity);
    // Secondary hash
    // unsigned int stepSize = secondaryHash(key.row, key.col, table->capacity);

    unsigned int originalIndex = index; // Variables for quadratic probing
    unsigned int i = 1;

    // Sum calclation - collision handling
    while (table->entries[index].occupied) {
        table->collisionCount++;
        if (table->entries[index].key.row == row && table->entries[index].key.col == col) { // Check if the product is to be summed
            table->entries[index].value += value;
            return;
        }
        // index = (index + 1) % table->capacity; // Linear probing
        index = (originalIndex + i * i) % table->capacity; // Quadratic probing
        i++;
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

// Function for merging two HashTables together
void mergeHashTables(HashTable *globalTable, HashTable *localTable) {
    for (int i = 0; i < localTable->capacity; i++) {
        if (localTable->entries[i].occupied) {  // Add up the values of with the same keys
            HashKey key = localTable->entries[i].key;
            double value = localTable->entries[i].value;
            hashTableInsert(globalTable, key.row, key.col, value);
        }
    }
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

/************************* Hash table functions for fine-grained locking **************************/

// Function to return the lock of a given hash key
int getLockIndex(unsigned int hashIndex, int lockCount) {
    return hashIndex % lockCount;
}

// Function to initialize a fine-grained locked hash table
void initHashTableLock(HashTable *table, int capacity) {
    table->capacity = capacity;
    table->size = 0;
    table->entries = (HashEntry *)calloc(capacity, sizeof(HashEntry));
    table->collisionCount = 0;

    // Initialize fine-grained locks
    table->lockCount = 64; // Number of locks (choose an appropriate number)
    table->locks = (pthread_mutex_t *)malloc(table->lockCount * sizeof(pthread_mutex_t));
    for (int i = 0; i < table->lockCount; i++) {
        pthread_mutex_init(&table->locks[i], NULL);
    }
}

void destroyHashTable(HashTable *table) {
    for (int i = 0; i < table->lockCount; i++) {
        pthread_mutex_destroy(&table->locks[i]);
    }
    free(table->locks);
    free(table->entries);
}

// Function for resizing hash table with fine-grained locks
void resizeHashTableLock(HashTable *table) {
    int oldCapacity = table->capacity;
    HashEntry *oldEntries = table->entries;
    
     // Free old locks and initialize new ones
    if (table->locks != NULL) { // Check if locks are already allocated
        for (int i = 0; i < table->lockCount; i++) {
            pthread_mutex_destroy(&table->locks[i]);
        }
        free(table->locks);
    }

    table->lockCount = 64; // Adjust the number of locks as needed
    table->locks = (pthread_mutex_t *)malloc(table->lockCount * sizeof(pthread_mutex_t));
    for (int i = 0; i < table->lockCount; i++) {
        pthread_mutex_init(&table->locks[i], NULL);
    }

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

// Insert or update an entry in the hash table
void hashTableInsertLock(HashTable *table, int row, int col, double value) {
    if (table->size > table->capacity * 0.7) { // Resize table incase 70% is full
        printf("I> Resized Hash table\n");
        clock_t resizeStart, resizeEnd; 
        resizeStart = clock();
        resizeHashTableLock(table);
        resizeEnd = clock();
        printf("I> Resize execution time: %f seconds\n", ((double) (resizeEnd - resizeStart)) / CLOCKS_PER_SEC ); 
    }

    HashKey key = { row, col };
    // Primary hash
    // unsigned int index = hash(key, table->capacity);
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);
    // unsigned int index = murmurHash3((uint32_t) key.row, (uint32_t) key.col, (uint32_t) table->capacity);
    // Secondary hash
    // unsigned int stepSize = secondaryHash(key.row, key.col, table->capacity);

    unsigned int originalIndex = index; // Variables for quadratic probing
    unsigned int i = 1;

    // Determine the lock index for fine-grained locking
    int lockIndex = getLockIndex(index, table->lockCount);

    pthread_mutex_lock(&table->locks[lockIndex]); // Lock the specific portion of the table

    // Sum calclation - collision handling
    while (table->entries[index].occupied) {
        table->collisionCount++;
        if (table->entries[index].key.row == row && table->entries[index].key.col == col) { // Check if the product is to be summed
            table->entries[index].value += value;
            pthread_mutex_unlock(&table->locks[lockIndex]); // Unlock after modification
            return;
        }
        // index = (index + 1) % table->capacity; // Linear probing
        index = (originalIndex + i * i) % table->capacity; // Quadratic probing
        i++;
    }

    table->entries[index].key = key;
    table->entries[index].value = value;
    table->entries[index].occupied = true;
    table->size++;

    pthread_mutex_unlock(&table->locks[lockIndex]); // Unlock after insertion
}

/************************* Testing reasons **************************/

void printDenseMatrix(SparseMatrixCOO *matrix) {
    int M = matrix->M;
    int N = matrix->N;
    
    // Create a temporary dense matrix initialized with zeros
    double denseMatrix[M][N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            denseMatrix[i][j] = 0.0;
        }
    }

    // Populate the dense matrix with values from the sparse matrix
    for (int k = 0; k < matrix->nnz; k++) {
        int row = matrix->I[k];
        int col = matrix->J[k];
        double value = matrix->val[k];
        denseMatrix[row][col] = value;
    }

    // Print the dense matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (denseMatrix[i][j] == 0.0) {
                printf("* ");
            } else {
                printf("%.2f ", denseMatrix[i][j]);
            }
        }
        printf("\n");
    }
}

// // Define a function pointer type for functions with no arguments and no return value
// typedef void (*FuncNoArgs)(void);

// // Function to measure the execution time of a function
// void measureExecutionTime(const char *message, FuncNoArgs func) {
//     clock_t start, end;
//     double executionTime;

//     start = clock();  // Start the timer

//     func();  // Call the function

//     end = clock();  // End the timer
//     executionTime = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calculate elapsed time

//     printf("%s execution time: %f seconds\n", message, executionTime);
// }