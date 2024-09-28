#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCSR A_csr, SparseMatrixCSR B_csr) {
    if (A_csr.N != B_csr.M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    HashTable *table = createHashTable(3*A_csr.nz); // Initial estimation for size (Greater initial est . Less hash table resizes . Greater chance of getting segfault)
    
    // Perform multiplication CSR
    for (int i = 0; i < A_csr.M; i++) { // Iterate over rows of A
        for (int j = A_csr.I_ptr[i]; j < A_csr.I_ptr[i + 1]; j++) { // Iterate over non-zeros in row i of A
            int aCol = A_csr.J[j]; // Column index in A
            double aVal = A_csr.val[j]; // Value in A

            for (int k = B_csr.I_ptr[aCol]; k < B_csr.I_ptr[aCol + 1]; k++) { // Iterate over non-zeros in column aCol of B
                int bRow = B_csr.I_ptr[k]; // Row index in B
                int bCol = B_csr.J[k]; // Column index in B
                double bVal = B_csr.val[k]; // Value in B
                double cVal = aVal * bVal;

                // Insert into hash table
                hashTableInsert(table, i, bCol, cVal, false);
            }
        }
    }

    Timer DOCtoCOOtime;
    startTimer(&DOCtoCOOtime);

    SparseMatrixCOO C = hashTableToSparseMatrix(table, A_csr.M, B_csr.N);

    stopTimer(&DOCtoCOOtime);

    printf("<I> Hash Table collision count: %d\n", table->collisionCount);
    printElapsedTime(&DOCtoCOOtime, "<I> DOK to COO conversion");

    // Free allocated memory
    freeCSRMatrix(&A_csr);
    freeCSRMatrix(&B_csr);
    freeHashTable(table);  

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
    if (argc == 4 && strcmp(argv[3], "--pprint") == 0) {
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

    Timer totalTimer;
    startTimer(&totalTimer);

    // Multiply A and B csr and return C coo
    C = multiplySparseMatrix(A_csr, B_csr);

    stopTimer(&totalTimer);
    // Print the result
    printSparseMatrix("C", &C, true);

    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&C);
    }

    printElapsedTime(&totalTimer,"Total multiplication");

    // Free the memory
    freeSparseMatrix(&C);

    return 0;
}
