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

SparseMatrixCOO computeGraphMinor(SparseMatrixCSR *A_csr, SparseMatrixCSR *Omega_csr, int numClusters) {
    // Initialize the memoization hash table
    HashTable *memo = createHashTable(4 * A_csr->nz);

    int numThreads = 4;

    // Set the number of threads
    omp_set_num_threads(numThreads);

    HashTable *tables[numThreads];
    for (int i = 0; i < numThreads; i++) {
        tables[i] = createHashTable(4 * A_csr->nz);
    }

    // Parallel region
    #pragma omp parallel shared(tables, A_csr, Omega_csr)
    {
        int thread_id = omp_get_thread_num();
        HashTable *local_table = tables[thread_id];

        // Compute the row start and end for each thread
        int rows_per_thread = A_csr->M / numThreads;
        int row_start = thread_id * rows_per_thread;
        int row_end = (thread_id == numThreads - 1) ? A_csr->M : (thread_id + 1) * rows_per_thread;

        // Process each row assigned to this thread
        for (int row = row_start; row < row_end; row++) {
            int start = A_csr->I_ptr[row];
            int end = A_csr->I_ptr[row + 1];

            // Iterate over non-zero elements in this row
            for (int i = start; i < end; i++) {
                int node_connect = A_csr->J[i];
                double node_weight = A_csr->val[i];

                // Get the clusters for the current node and the connected node
                int cluster = Omega_csr->J[row];
                int cluster_connect = Omega_csr->J[node_connect];

                // Adjust the weight if both nodes belong to the same cluster
                if (cluster == cluster_connect) {
                    node_weight /= 2;
                }

                // Use the local hash table to store the result for (c_u, c_v)
                hashTableInsert(local_table, cluster, cluster_connect, node_weight, false);
            }
        }
    }

    // Combine the results from the local hash tables into the final matrix
    SparseMatrixCOO M = hashTablesToSparseMatrix(tables, numThreads, numClusters, numClusters);

    // Free the hash table memory
    freeHashTable(memo);
    for (int i = 0; i < numThreads; i++) {
        freeHashTable(tables[i]);
    }

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

    SparseMatrixCSR A_csr, Omega_csr;
    // Convert COO matrices to CSR format
    A_csr = COOtoCSR(A);
    Omega_csr = COOtoCSR(Omega);

    // Step 2:
    // Compute the graph minor's adjacency matrix M
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    SparseMatrixCOO M = computeGraphMinor(&A, &Omega, numClusters);

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
    freeSparseMatrix(&M);

    return EXIT_SUCCESS;
}