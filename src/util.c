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

typedef struct HashEntry{
    HashKey key;
    double value;
    struct HashEntry *next; // Pointer to the next entry in the linked list 
} HashEntry;

typedef struct {
    HashEntry **buckets; // Array of pointers to linked lists of entries
    int capacity;       // Total number of buckets 
    int size;           // Number of elements in the hash table 
    int collisionCount; // Number of collisions encountered
} HashTable;

/******************************** Matrix data structs ***************************** */

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

/******************************** Init sparse matrix functions ***************************** */

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

/******************************** Print sparse matrix functions ***************************** */

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

// Function to pretty print a matrix for debugging
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

    // Determine the width for each cell
    int cellWidth = 8;

    // Print the top border
    printf("+");
    for (int j = 0; j < N; j++) {
        printf("%*s", cellWidth + 1, "--------");
    }
    printf("+\n");

    // Print the dense matrix with borders and alignment
    for (int i = 0; i < M; i++) {
        printf("|");
        for (int j = 0; j < N; j++) {
            if (denseMatrix[i][j] == 0.0) {
                printf("%*s", cellWidth, "*");
            } else {
                printf("%*.2f", cellWidth, denseMatrix[i][j]);
            }
            printf(" ");
        }
        printf("|\n");
    }

    // Print the bottom border
    printf("+");
    for (int j = 0; j < N; j++) {
        printf("%*s", cellWidth + 1, "--------");
    }
    printf("+\n");
}

int readSparseMatrix(const char *filename, SparseMatrixCOO *mat) {
    int ret = mm_read_unsymmetric_sparse(filename, &mat->M, &mat->N, &mat->nnz, &mat->val, &mat->I, &mat->J);
    if (ret!=0) {
        fprintf(stderr, "\tFailed to read the matrix from file %s\n", filename);
        return ret;
    }
    return 0;
}

/******************************** Hash table functions ***************************** */

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

// Function to create a new hash entry
HashEntry *createHashEntry(int row, int col, double value) {
    HashEntry *entry = (HashEntry *)malloc(sizeof(HashEntry));
    entry->key.row = row;
    entry->key.col = col;
    entry->value = value;
    entry->next = NULL;
    return entry;
}

// Function to create a hash table
HashTable *createHashTable(int initialCapacity) {
    HashTable *table = (HashTable *)malloc(sizeof(HashTable));
    table->capacity = initialCapacity;
    table->size = 0;
    table->collisionCount = 0;
    table->buckets = (HashEntry **)calloc(initialCapacity, sizeof(HashEntry *));
    return table;
}

void resizeHashTable(HashTable *table);

// Insert or update an entry in the hash table
void hashTableInsert(HashTable *table, int row, int col, double value) {
    // Resize the table if load factor exceeds 65%
    if (table->size > table->capacity * 0.65) {
        printf("I> Resized Hash table\n");
        clock_t resizeStart, resizeEnd;
        resizeStart = clock();
        resizeHashTable(table);
        resizeEnd = clock();
        printf("I> Resize execution time: %f seconds\n", ((double)(resizeEnd - resizeStart)) / CLOCKS_PER_SEC);
    }

    HashKey key = {row, col};
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);
    HashEntry *current = table->buckets[index];

    // Traverse the linked list to find if the key already exists
    while (current) {
        if (current->key.row == row && current->key.col == col) {
            current->value += value;  // Update the existing value
            return;
        }
        current = current->next;
    }

    // Create a new entry and insert it at the head of the linked list
    HashEntry *newEntry = createHashEntry(row, col, value);
    newEntry->next = table->buckets[index];
    table->buckets[index] = newEntry;
    table->size++;
    table->collisionCount++;
}

// Function to resize the hash table
void resizeHashTable(HashTable *table) {
    int oldCapacity = table->capacity;
    HashEntry **oldBuckets = table->buckets;

    // Double the capacity and reallocate buckets
    table->capacity *= 2;
    table->buckets = (HashEntry **)calloc(table->capacity, sizeof(HashEntry *));
    table->size = 0;

    // Rehash all entries from the old table into the new one
    for (int i = 0; i < oldCapacity; i++) {
        HashEntry *entry = oldBuckets[i];
        while (entry) {
            HashEntry *next = entry->next;
            unsigned int index = fnv1aHash(entry->key.row, entry->key.col, table->capacity);
            entry->next = table->buckets[index];
            table->buckets[index] = entry;
            table->size++;
            entry = next;
        }
    }

    free(oldBuckets);
}

// Function to free the memory allocated for the hash table
void freeHashTable(HashTable *table) {
    for (int i = 0; i < table->capacity; i++) {
        HashEntry *entry = table->buckets[i];
        while (entry) {
            HashEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(table->buckets);
    free(table);
}


// Function to convert hash table to sparse matrix COO format
SparseMatrixCOO hashTableToSparseMatrix(HashTable *table, int M, int N) {
    SparseMatrixCOO C;
    // Initialize the SparseMatrixCOO with the dimensions and estimated size
    initSparseMatrix(&C, M, N, table->size);  // table->size gives the count of entries

    int index = 0;  // Index for inserting into the COO arrays
    // Iterate over each bucket in the hash table
    for (int i = 0; i < table->capacity; i++) {
        HashEntry *entry = table->buckets[i];  // Get the head of the linked list for this bucket
        // Traverse the linked list in the current bucket
        while (entry != NULL) {
            // Populate the COO matrix with the key (row, col) and value from each entry
            C.I[index] = entry->key.row;
            C.J[index] = entry->key.col;
            C.val[index] = entry->value;
            index++;
            entry = entry->next;  // Move to the next node in the linked list
        }
    }
    C.nnz = index;  // Set the actual number of non-zero elements
    return C;
}
