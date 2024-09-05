#include <stdio.h>

#include "util.c"

// Function to multiply sparse matrices
SparseMatrixCOO multiplySparseMatrix(SparseMatrixCSR *A_csr, SparseMatrixCSR *B_csr) {
    if (A_csr->N != B_csr->M) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    HashTable *resultTable = createHashTable(A_csr->nz  + B_csr->nz); // Initial estimation for size (Greater initial est -> Less hash table resizes -> Greater chance of getting segfault)
    
    // Perform multiplication with accumulator
    for (int i = 0; i < A_csr->M; i++) {  // Iterate over rows of A
        Accumulator *acc = createAccumulator(B_csr->N);  // Initialize a new accumulator for the row

        for (int j = A_csr->I_ptr[i]; j < A_csr->I_ptr[i + 1]; j++) {  // Iterate over non-zeros in row i of A
            int aCol = A_csr->J[j];       // Column index in A
            double aVal = A_csr->val[j];  // Value in A

            // Mark allowed columns based on the non-zeros in B
            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) {
                int bCol = B_csr->J[k];  // Column index in B
                setAllowed(acc, bCol);   // Allow this column
            }

            // Perform multiplication and accumulate results
            for (int k = B_csr->I_ptr[aCol]; k < B_csr->I_ptr[aCol + 1]; k++) {
                int bCol = B_csr->J[k];       // Column index in B
                double bVal = B_csr->val[k];  // Value in B
                double cVal = aVal * bVal;

                // Accumulate the value in the accumulator
                accumulatorInsert(acc, i, bCol, cVal);
            }
        }

        // Transfer accumulated results to the final hash table
        for (int j = 0; j < B_csr->N; j++) {
            if (acc->allowed[j]) {
                double finalValue = accumulatorRemove(acc, i, j);
                if (finalValue != 0) {
                    hashTableInsert(resultTable, i, j, finalValue);  // Insert into the final hash table
                }
            }
        }

        // Clean up the accumulator
        destroyAccumulator(acc);
    }

    Timer DOCtoCOOtime;
    startTimer(&DOCtoCOOtime);

    SparseMatrixCOO C = hashTableToSparseMatrix(resultTable, A_csr->M, B_csr->N);

    stopTimer(&DOCtoCOOtime);

    printf("<I> Hash Table collision count: %d\n", resultTable->collisionCount);
    printElapsedTime(&DOCtoCOOtime, "<I> DOK to COO conversion");

    // Free allocated memory
    freeCSRMatrix(A_csr);
    freeCSRMatrix(B_csr);
    freeHashTable(resultTable);  

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

    Timer totalTimer;
    startTimer(&totalTimer);

    // Multiply A and B csr and return C coo
    C = multiplySparseMatrix(&A_csr, &B_csr);

    stopTimer(&totalTimer);
    // Print the result
    printSparseMatrix("C", &C, true);

    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&C);
    }

    printElapsedTime(&totalTimer,"Total multiplication");

    // Free the memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&B);
    freeSparseMatrix(&C);

    return 0;
}
