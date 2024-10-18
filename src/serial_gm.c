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
    for (int i = 0; i < A->nnz; i++) {
        int node = A->I[i];
        int node_connect = A->J[i];
        double node_weight = A->val[i];

        // Print the current node and its connection
        printf("Processing A[%d]: (node: %d, connects to: %d, weight: %lf)\n", i, node, node_connect, node_weight);

        int cluster = Omega->J[node];
        int cluster_connect = Omega->J[node_connect];

        // Ensure we only process the "upper triangular" part by enforcing an order
        if (cluster != cluster_connect) {
            // Find the clusters that nodes u and v belong to
            

            // Print the corresponding clusters for the current node
            printf("Node %d belongs to cluster %d, and node %d belongs to cluster %d\n", node, cluster, node_connect, cluster_connect);

            // Use hash table to store the result for (cluster, cluster_connect)
            printf("Inserting into hash table: (cluster: %d, cluster_connect: %d, weight: %lf)\n", cluster, cluster_connect, node_weight);

        } else {
            printf("Symmetric connection; Adding half weight : (node: %d, connects to: %d)\n", node, node_connect);

            node_weight /= 2;
        }

        hashTableInsert(memo, cluster, cluster_connect, node_weight, false);
    }

    printHashTable(memo);

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

    
    // Pretty print the dense matrix if -pprint flag is provided

    Omega.I[0] = 0;  Omega.J[0] = 1;  Omega.val[0] = 1.0;
    Omega.I[1] = 1;  Omega.J[1] = 1;  Omega.val[1] = 1.0;
    Omega.I[2] = 2;  Omega.J[2] = 2;  Omega.val[2] = 1.0;
    Omega.I[3] = 3;  Omega.J[3] = 0;  Omega.val[3] = 1.0;

    printSparseMatrix("Ω", &Omega, true);

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