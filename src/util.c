#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>

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
    /*
        HashTable
        +-------------------------------------------+
        |  []  []  []  []  []  []  []  []  []  []   |
        +-------------------------------------------+
          |   
          v
        bucket
          |
          v
        +---+    +---+
        | o | -> | o | -> NULL
        +---+    +---+
        entry
    */
    HashEntry **buckets; // Array of pointers to linked lists of entries
    int capacity;       // Total number of buckets / linked lists
    int size;           // Number of elements in the hash table 
    int collisionCount; // Number of collisions encountered
    pthread_mutex_t *bucketLocks;
    pthread_mutex_t *resizeLock;
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

typedef struct SparseMatrixCSR{

    /*
            N , J
    +-------------------+
    | *                 |
    |    *    *         |
    |    *  *           | M, I_ptr
    |          *  *   * |
    |             *     |
    |      *   *     *  |
    +-------------------+

    nnz = *
    I_ptr = Row pointer (size M+1)
    J[nnz] = Column indices of non-zeros
    val[nnz] = Non-zero values

    Legend:
    M = Rows
    N = Columns
    I_ptr = Row pointers (array of size M+1)
    J = Column indices (array of size nnz)
    val = Non-zero values (array of size nnz)
   */

    int *I_ptr;
    int *J;
    double *val;

    int M;
    int N;
    int nz;
} SparseMatrixCSR;

/******************************** Thread data structs ***************************** */

// Struct to hold thread information 
typedef struct {
    int thread_id;                  // Id for each thread
    SparseMatrixCOO *A;             // Matrix A
    SparseMatrixCOO *B;             // Matrix B
    SparseMatrixCSR *A_csr;         // Matrix A in CSR format
    SparseMatrixCSR *B_csr;         // Matrix B in CSR format
    int start;                      // Thread workload start
    int end;                        // Thread workload end
    HashTable *table;               // Hash table
    pthread_mutex_t *bucketLocks;   // Mutexes for each of the table's buckets
    unsigned int index;             // Hash index for each thread
} ThreadData;

/******************************** Misc functions ***************************** */

typedef struct {
    clock_t start_time;
    clock_t end_time;
} Timer;

// Start the timer
void startTimer(Timer* timer) {
    timer->start_time = clock();
}

// Stop the timer
void stopTimer(Timer* timer) {
    timer->end_time = clock();
}

// Print elapsed time message
void printElapsedTime(Timer* timer, char *message) {
    double elapsed = (double)(timer->end_time - timer->start_time) / CLOCKS_PER_SEC;
    printf("%s Execution time: %.6f seconds\n", message, elapsed);
}

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
void hashTableInsert(HashTable *table, int row, int col, double value, bool resize) {
    if (resize && table->size > table->capacity * 0.65) { // Resize table incase 70% is full
        Timer resizeTime;
        startTimer(&resizeTime);

        resizeHashTable(table);

        stopTimer(&resizeTime);
        printElapsedTime(&resizeTime, "<I> Resized Hash Table! "); 
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

// Remove an entry from the hash table
void hashTableRemove(HashTable *table, int row, int col) {
    HashKey key = {row, col};
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);
    HashEntry *current = table->buckets[index];
    HashEntry *prev = NULL;

    // Traverse the linked list to find the entry to remove
    while (current) {
        if (current->key.row == row && current->key.col == col) {
            // Found the entry to remove
            if (prev) {
                // Not the head of the list
                prev->next = current->next;
            } else {
                // Head of the list
                table->buckets[index] = current->next;
            }
            free(current);  // Free the memory allocated for the entry
            table->size--;
            return;
        }
        prev = current;
        current = current->next;
    }
}

// Retrieves the entry from the hash table based on row and column
HashEntry *hashTableGet(HashTable *table, int row, int col) {
    HashKey key = {row, col};
    unsigned int index = fnv1aHash(key.row, key.col, table->capacity);
    HashEntry *current = table->buckets[index];

    // Traverse the linked list to find the entry with the matching key
    while (current) {
        if (current->key.row == row && current->key.col == col) {
            return current;  // Entry found
        }
        current = current->next;
    }
    return NULL;  // Entry not found
}

// Function to merge multiple private hash tables into a global hash table
void mergeHashTables(HashTable *tables[], HashTable *globalTable, int numTables, bool resize) {

    // Merge each private hash table into the global hash table
    for (int i = 0; i < numTables; i++) {
        // Iterate over each bucket in the private table
        for (int j = 0; j < tables[i]->capacity; j++) {
            HashEntry *entry = tables[i]->buckets[j];

            // Iterate over each entry in the linked list at this bucket
            while (entry) {
                int row = entry->key.row;
                int col = entry->key.col;
                double value = entry->value;

                // Insert the entry into the global table
                hashTableInsert(globalTable, row, col, value, resize);

                entry = entry->next;
            }
        }
    }
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
SparseMatrixCOO hashTableToSparseMatrix(HashTable *table, int numRows, int numCols) {
    // Ensure the table is valid
    if (!table || !table->buckets) {
        fprintf(stderr, "<E> Hash table is null or buckets are uninitialized.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the SparseMatrixCOO with the dimensions and estimated size
    SparseMatrixCOO C;
    initSparseMatrix(&C, numRows, numCols, table->size);

    printf("<D> Hash table size before conversion = %d\n", table->size);

    if (!C.I || !C.J || !C.val) {
        fprintf(stderr, "Error: Failed to allocate memory for sparse matrix COO format.\n");
        exit(EXIT_FAILURE);
    }

    int index = 0;                                      // Index for inserting into the COO arrays
    for (int i = 0; i < table->capacity; i++) {         // Iterate over each bucket in the hash table
        HashEntry *entry = table->buckets[i];           // Get the head of the linked list for this bucket
        while (entry) {                                 // Traverse the linked list in the current bucket
            // Ensure index is within bounds
            if (index >= C.nnz) {
                fprintf(stderr, "<E> Index out of bounds. nnz mismatch.\n");
                exit(EXIT_FAILURE);
            }

            // Populate the COO matrix with the key (row, col) and value from each entry
            C.I[index] = entry->key.row;
            C.J[index] = entry->key.col;
            C.val[index] = entry->value;

            // Move to the next node in the linked list
            entry = entry->next;
            index++;
        }
    }

    // Final check to ensure all non-zero elements are accounted for
    if (index != C.nnz) {
        fprintf(stderr, "<E> COO matrix size mismatch. Expected nnz=%d, but got %d.\n", C.nnz, index);
        exit(EXIT_FAILURE);
    }

    C.nnz = index;  // Set the actual number of non-zero elements
    return C;
}

/******************************** Locked Hash table functions ***************************** */

#define MAX_RETRIES 10
#define RETRY_DELAY_US 1

// Insert or update an entry in the hash table with thread safety
void hashTableInsert_L(ThreadData *data, HashTable *table, int row, int col, double value, bool resize) {

    data->index = fnv1aHash(row, col, table->capacity); // Each thread calculates its own hash index to insert into
    int retries = 0;
    bool success = false;

    while (retries < MAX_RETRIES){
        // Try to lock the specific bucket
        if(pthread_mutex_trylock(&table->bucketLocks[data->index]) == 0) {
            // Lock acquired succesfully
            success = true;
            break;
        }
        // Lock not aquired, try again
        printf("Thread %d tried accessing a busy bucket, retrying...\n",data->thread_id);
        retries++;
        usleep(RETRY_DELAY_US * pow(2, retries)); // Exponential backoff
    } 

    if (!success) {
        // Max retries reached, failed to acquire lock
        fprintf(stderr, "<E> Thread %d failed to acquire lock for bucket %d after %d retries\n", data->thread_id, data->index, MAX_RETRIES);
        exit(EXIT_FAILURE); // Exit the function without further processing
    }

    // Proceed with inserting or updating the entry since the lock was acquired
    HashEntry *current = table->buckets[data->index];

    // Traverse the linked list to find if the key already exists
    while (current) {
        if (current->key.row == row && current->key.col == col) {
            current->value += value;  // Update the existing value
            pthread_mutex_unlock(&table->bucketLocks[data->index]);
            return;
        }
        current = current->next;
    }

    // Create a new entry and insert it at the head of the linked list
    HashEntry *newEntry = createHashEntry(row, col, value);
    newEntry->next = table->buckets[data->index];
    table->buckets[data->index] = newEntry;
    __sync_fetch_and_add(&table->size, 1);  // Atomic increment to prevent race conditions
    table->collisionCount++;

    // Unlock the bucket after modification
    pthread_mutex_unlock(&table->bucketLocks[data->index]);
}

// // Function to resize the hash table
// void resizeHashTable_L(HashTable *table) {
//     int oldCapacity = table->capacity;
//     HashEntry **oldBuckets = table->buckets;
//     pthread_mutex_t *oldBucketLocks = table->bucketLocks;

//     // Double the capacity and reallocate buckets and locks
//     table->capacity *= 2;
//     table->buckets = (HashEntry **)calloc(table->capacity, sizeof(HashEntry *));
//     table->bucketLocks = (pthread_mutex_t *)malloc(table->capacity * sizeof(pthread_mutex_t));
//     for (int i = 0; i < table->capacity; i++) {
//         pthread_mutex_init(&table->bucketLocks[i], NULL);
//     }
//     table->size = 0;

//     // Rehash all entries from the old table into the new one
//     for (int i = 0; i < oldCapacity; i++) {
//         pthread_mutex_lock(&oldBucketLocks[i]); // Lock old buckets during transfer
//         HashEntry *entry = oldBuckets[i];
//         while (entry) {
//             HashEntry *next = entry->next;
//             unsigned int index = fnv1aHash(entry->key.row, entry->key.col, table->capacity);

//             // Insert into the new buckets without locking (no other threads access the new table yet)
//             entry->next = table->buckets[index];
//             table->buckets[index] = entry;
//             table->size++;
//             entry = next;
//         }
//         pthread_mutex_unlock(&oldBucketLocks[i]); // Unlock after transfer
//     }

//     // Clean up old locks and buckets
//     for (int i = 0; i < oldCapacity; i++) {
//         pthread_mutex_destroy(&oldBucketLocks[i]);
//     }
//     free(oldBucketLocks);
//     free(oldBuckets);
// }

// Function to free the memory allocated for the hash table
void freeHashTable_L(HashTable *table) {
    for (int i = 0; i < table->capacity; i++) {
        pthread_mutex_destroy(&table->bucketLocks[i]); // Destroy each bucket lock
        HashEntry *entry = table->buckets[i];
        while (entry) {
            HashEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(table->bucketLocks);
    free(table->buckets);
    free(table);
}

// Function to create a hash table with error checking
HashTable *createHashTable_L(int initialCapacity) {
    // Allocate memory for the hash table structure
    HashTable *table = (HashTable *)malloc(sizeof(HashTable));
    if (table == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for hash table.\n");
        exit(EXIT_FAILURE);
    }
    
    table->capacity = initialCapacity;
    table->size = 0;
    table->collisionCount = 0;
    table->buckets = (HashEntry **)calloc(initialCapacity, sizeof(HashEntry *));
    if (table->buckets == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for hash table buckets.\n");
        free(table);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for bucket locks
    table->bucketLocks = (pthread_mutex_t *)malloc(initialCapacity * sizeof(pthread_mutex_t));
    if (table->bucketLocks == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for bucket locks.\n");
        free(table->buckets);
        free(table);
        exit(EXIT_FAILURE);
    }

    // Initialize each mutex lock
    for (int i = 0; i < initialCapacity; i++) {
        if (pthread_mutex_init(&table->bucketLocks[i], NULL) != 0) {
            fprintf(stderr, "Error: Failed to initialize mutex lock.\n");
            // Cleanup previously initialized locks
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&table->bucketLocks[j]);
            }
            free(table->bucketLocks);
            free(table->buckets);
            free(table);
            exit(EXIT_FAILURE);
        }
    }

    return table;
}

/******************************** CSR functions ***************************** */

// Function to initialize a CSR matrix
void initSparseMatrixCSR(SparseMatrixCSR *mat, int M, int N, int nz) {
    mat->M = M;
    mat->N = N;
    mat->nz = nz;
    
    mat->I_ptr = (int *)malloc((M + 1) * sizeof(int));
    mat->J = (int *)malloc(nz * sizeof(int));
    mat->val = (double *)malloc(nz * sizeof(double));
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

// Converts COO to CSR
SparseMatrixCSR COOtoCSR(SparseMatrixCOO A) {
    SparseMatrixCSR output;
    initSparseMatrixCSR(&output, A.M, A.N, A.nnz);

    // Initialize row pointers
    for (int i = 0; i <= output.M; i++) {
        output.I_ptr[i] = 0;
    }

    // Count the number of non-zero elements per row
    for (int i = 0; i < A.nnz; i++) {
        output.I_ptr[A.I[i] + 1]++;
    }

    // Accumulate row pointers
    for (int i = 0; i < output.M; i++) {
        output.I_ptr[i + 1] += output.I_ptr[i];
    }

    // Populate J and val arrays
    for (int i = 0; i < A.nnz; i++) {
        int row = A.I[i];
        int destIndex = output.I_ptr[row]++;
        output.J[destIndex] = A.J[i];
        output.val[destIndex] = A.val[i];
    }

    // Fix the row pointer array by shifting pointers
    for (int i = output.M; i > 0; i--) {
        output.I_ptr[i] = output.I_ptr[i - 1];
    }
    output.I_ptr[0] = 0;

    return output;
}

// Function to free the memory allocated for a SparseMatrixCSR
void freeCSRMatrix(SparseMatrixCSR *mat) {
    free(mat->I_ptr);
    free(mat->J);
    free(mat->val);
}

// Converts CSR to COO
SparseMatrixCOO CSRtoCOO(SparseMatrixCSR A) {
    SparseMatrixCOO output;
    initSparseMatrix(&output, A.M, A.N, A.nz);

    int index = 0;
    
    for (int i = 0; i < A.M; i++) {
        for (int j = A.I_ptr[i]; j < A.I_ptr[i + 1]; j++) {
            output.I[index] = i;
            output.J[index] = A.J[j];
            output.val[index] = A.val[j];
            index++;
        }
    }

    return output;
}