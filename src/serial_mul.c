#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCOO *A, SparseMatrixCOO *B, SparseMatrixCSR *A_csr, SparseMatrixCSR *B_csr) {
    if (A->N != B->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    HashTable *table = createHashTable(A->nnz + B->nnz); // Initial estimation for size (Greater initial est -> Less hash table resizes -> Greater chance of getting segfault)

    // for (int i = 0; i < A->nnz; i++) {
    //     for (int j = 0; j < B->nnz; j++) {
    //         if (A->J[i] == B->I[j]) {
    //             int row = A->I[i];
    //             int col = B->J[j];
    //             double value = A->val[i] * B->val[j];
    //             hashTableInsert(table, row, col, value);
    //         }
    //     }
    // }
    // Perform multiplication
    for (int i = 0; i < A_csr->M; i++) { // Iterate over rows of A
        for (int j = A_csr->I_ptr[i]; j < A_csr->I_ptr[i + 1]; j++) { // Iterate over non-zeros in row i of A
            int aCol = A_csr->J[j]; // Column index in A
            double aVal = A_csr->val[j]; // Value in A

            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) { // Iterate over non-zeros in column aCol of B
                int bRow = B_csr->I_ptr[k]; // Row index in B
                int bCol = B_csr->J[k]; // Column index in B
                double bVal = B_csr->val[k]; // Value in B
                double cVal = aVal * bVal;

                // Insert into hash table
                hashTableInsert(table, i, bCol, cVal);
            }
        }
    }

    clock_t hashStart, hashEnd;
    double hashToCOOTime;
    hashStart = clock();

    SparseMatrixCOO C = hashTableToSparseMatrix(table, A->M, B->N);

    hashEnd = clock();
    hashToCOOTime = ((double) (hashEnd - hashStart)) / CLOCKS_PER_SEC;

    printf("I> Hash Table collision count: %d\n", table->collisionCount);
    printf("I> DOK to COO conversion execution time: %f seconds\n", hashToCOOTime);

    freeHashTable(table);  // Free the hash table entries

    return C;
}

int main(int argc, char *argv[]) {

    // Input the matrix filename arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file_A> <matrix_file_B> [-pprint]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file_A = argv[1];
    const char *matrix_file_B = argv[2];
    bool pprint_flag = false;

    // Check for the -pprint flag
    if (argc == 4 && strcmp(argv[3], "-pprint") == 0) {
        pprint_flag = true;
    }

    SparseMatrixCOO A, B, C;

    // Read from files
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }
    
    if (readSparseMatrix(matrix_file_B, &B) != 0) {
        freeSparseMatrix(&A); // Free A if B read fails
        return EXIT_FAILURE;
    }

    SparseMatrixCSR A_csr, B_csr;
    // Convert COO matrices to CSR format
    A_csr = COOtoCSR(A);
    B_csr = COOtoCSR(B);

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // Multiply A and B csr and return C coo
    C = multiplySparseMatrix(&A, &B, &A_csr, &B_csr);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Print the result
    printSparseMatrix("C", &C, true);

    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&C);
    }

    printf("\nI> Total multiplication execution time: %f seconds\n", cpu_time_used);

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}
