#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include "mmio.h"
#include "util.c"

#define MAX_THREADS 4

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
    pthread_mutex_t lock;
} HashTable;

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

// Hash function for HashKey
unsigned int hash(HashKey key, int capacity) {
    unsigned int hashValue = key.row * 31 + key.col;
    return hashValue % capacity;
}
// Initialize the hash table
void initHashTable(HashTable *table, int capacity) {
    table->capacity = capacity;
    table->size = 0;
    table->entries = (HashEntry *)calloc(capacity, sizeof(HashEntry));
    pthread_mutex_init(&table->lock, NULL);
}

void resizeHashTable(HashTable *table);

// Insert or update an entry in the hash table
void hashTableInsert(HashTable *table, int row, int col, double value) {
    
    if (table->size > table->capacity * 0.7) {
        resizeHashTable(table);
    }
    
    HashKey key = { row, col };
    unsigned int index = hash(key, table->capacity);

    // Linear probing for collision resolution
    while (table->entries[index].occupied) {
        if (table->entries[index].key.row == row && table->entries[index].key.col == col) {
            table->entries[index].value += value;
            return;
        }
        index = (index + 1) % table->capacity; // LInear probing
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
            // LOCK
            pthread_mutex_lock(&table->lock);
            hashTableInsert(table, oldEntries[i].key.row, oldEntries[i].key.col, oldEntries[i].value);
            // UNLOCK
            pthread_mutex_unlock(&table->lock);
        }
    }

    free(oldEntries);
}

// Convert hash table to sparse matrix COO format
SparseMatrixCOO hashTableToSparseMatrix(HashTable *table) {
    SparseMatrixCOO C;
    initSparseMatrix(&C, 0, 0, table->size);
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

void* threadMultiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    SparseMatrixCOO *A = data->A;
    SparseMatrixCOO *B = data->B;
    HashTable *table = data->table;

    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < B->nnz; j++) {
            if (A->J[i] == B->I[j]) {
                int row = A->I[i];
                int col = B->J[j];
                double value = A->val[i] * B->val[j];
                hashTableInsert(table, row, col, value);
            }
        }
    }

    return NULL;
}

SparseMatrixCOO multiplySparseMatrixParallel(SparseMatrixCOO *A, SparseMatrixCOO *B, int numThreads) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    int initialCapacity = 1024; 
    HashTable table;
    initHashTable(&table, initialCapacity);

    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int chunkSize = (A->nnz + numThreads - 1) / numThreads;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].A = A;
        threadData[i].B = B;
        threadData[i].table = &table;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i + 1) * chunkSize;
        if (threadData[i].end > A->nnz) {
            threadData[i].end = A->nnz;
        }

        pthread_create(&threads[i], NULL, threadMultiply, &threadData[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    SparseMatrixCOO C = hashTableToSparseMatrix(&table);
    free(table.entries);  // Free the hash table entries

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
    
    if (readSparseMatrix(matrix_file_B, &B) != 0) {
        freeSparseMatrix(&A); // Free A if B read fails
        return EXIT_FAILURE;
    }


    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // Multiply A and B
    C = multiplySparseMatrixParallel(&A, &B, MAX_THREADS);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", cpu_time_used);

    // Print the result
    printSparseMatrix(&C, true);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}
