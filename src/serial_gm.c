#include <stdio.h>

#include "util.c"

// Function to randomly assign nodes to clusters and construct the Omega matrix
void constructOmegaMatrix(SparseMatrixCOO *Omega, int numNodes, int numClusters) {
    // Count the number of non-zero entries (nnz) of Omega
    int nnz = numNodes;

    // Initialize the Omega matrix
    initSparseMatrix(Omega, numNodes, numClusters, nnz);

    // Seed the random number generator
    srand(time(NULL));

    // Fill in the matrix
    for (int i = 0; i < numNodes; i++) {
        int cluster = rand() % numClusters;     // Randomly assign node i to a cluster
        Omega->I[i] = i;                        // Row index corresponds to the node
        Omega->J[i] = cluster;                  // Column index corresponds to the cluster
        Omega->val[i] = 1.0;                    // Value is 1 (indicating membership)
    }
}

SparseMatrixCOO computeGraphMinor(SparseMatrixCOO *A, SparseMatrixCOO *Omega, int numClusters) {
    // Initialize the hash table
    HashTable *memo = createHashTable(3 * A->nnz);

    // Iterate over each non-zero entry in the sparse matrix A
    for (int idx = 0; idx < A->nnz; idx++) {
        int A_row = A->I[idx];
        int A_col = A->J[idx];
        double A_val = A->val[idx];

        // Find the clusters that nodes u and v belong to
        int M_row = Omega->J[A_row];
        int M_col = Omega->J[A_col];

        // Use hash table to store the result for (c_u, c_v)
        hashTableInsert(memo, M_row, M_col, A_val, true);
    }

    // Convert the hash table into the sparse matrix M
    SparseMatrixCOO M = hashTableToSparseMatrix(memo, numClusters, numClusters);

    // Free the hash table memory
    freeHashTable(memo);

    return M;
}

int main (int argc, char *argv[]) {

    // Input the matrix filename arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file_A> <num_clusters> [-pprint]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file_A = argv[1];
    int numClusters = atoi(argv[2]);
    bool pprint_flag = false;

    // Check for the -pprint flag
    if (argc == 4 && strcmp(argv[3], "-pprint") == 0) {
        pprint_flag = true;
    }

    SparseMatrixCOO A, Omega, M;

    // Read from files
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }

    int numNodes = A.M;  // Assuming A is a square matrix, numNodes is the number of rows/columns

    // Step 1: 
    // Construct the Omega matrix with random cluster assignments
    constructOmegaMatrix(&Omega, numNodes, numClusters);

    printSparseMatrix("Ω", &Omega, true);
    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&Omega);
    }

    // Step 2:
    // Compute the graph minor's adjacency matrix M
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    M = computeGraphMinor(&A, &Omega, numClusters);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Results

    // Print the result matrix M
    printSparseMatrix("M",&M, true);
    // Pretty print the dense matrix if -pprint flag is provided
    if (pprint_flag) {
        printDenseMatrix(&M);
    }
    printf("\nI> Total execution time: %f seconds\n", cpu_time_used);
    
    // Free memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&Omega);
    freeSparseMatrix(&M);

    return EXIT_SUCCESS;
}