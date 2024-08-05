#include <stdio.h>
#include <stdbool.h>
#include "util.c"

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
} HashTable;

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