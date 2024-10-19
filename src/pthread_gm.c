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

// Function that each thread will run 
void *computeGraphMinorThread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    SparseMatrixCSR *A = data->A_csr;
    SparseMatrixCOO *Omega = data->B; // Omega representing cluster information
    HashTable *memo = data->table;

    // Process each row assigned to each thread
    for (int row = data->row_start; row < data->row_end; row++){

        int start = A->I_ptr[row];
        int end = A->I_ptr[row + 1];

        // Iterate over non zero elements in this row
        for (int i = start; i < end; i ++) {
            int node_connect = A->J[i];
            double node_weight = A->val[i];
            // row represents the current node

            int cluster = Omega->J[row];
            int cluster_connect = Omega->J[node_connect];
            
            if (cluster == cluster_connect) node_weight /= 2;
            // Use the local hash table to store the result for (c_u, c_v)
            hashTableInsert(memo, cluster, cluster_connect, node_weight, false);
        }

    }

    return NULL;
}

// Updated computeGraphMinor using fine-grained locking
SparseMatrixCOO computeGraphMinor(SparseMatrixCOO *A, SparseMatrixCOO *Omega, int numClusters) {
    int numThreads = 2;  // Set the number of threads
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    // Convert A from COO to CSR
    SparseMatrixCSR A_csr = COOtoCSR(*A);
    
    // Create array of hash tables to assign to each thread
    HashTable *tables[numThreads];

    // Assign rows of A to each thread
    int total_rows = A_csr.M;
    int base_rows = total_rows / numThreads;  // Base chunk size 
    int remainder_rows = total_rows % numThreads;  // Remainder rows to distribute evenly
    for (int i = 0; i < numThreads; i++) {
        tables[i] = createHashTable(3 * A_csr.nz); // Each thread gets its private hash table

        // Prepare the thread data
        threadData[i].thread_id = i;
        threadData[i].A_csr = &A_csr;
        threadData[i].B = Omega;
        threadData[i].row_start = i * base_rows + (i < remainder_rows ? i : remainder_rows);
        threadData[i].row_end = threadData[i].row_start + base_rows + (i < remainder_rows ? 1 : 0);
        threadData[i].table = tables[i];
        printf("Thread %d started processing rows from %d to %d\n", i, threadData[i].row_start, threadData[i].row_end);

        // Launch the thread
        pthread_create(&threads[i], NULL, computeGraphMinorThread, (void *)&threadData[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Convert the hash tables into the sparse matrix M
    SparseMatrixCOO M = hashTablesToSparseMatrix(tables, numThreads, numClusters, numClusters);

    // Free allocated memory
    for (int i = 0; i < numThreads; i++) {
        freeHashTable(tables[i]);
    }
    freeCSRMatrix(&A_csr);

    return M;
}

int main (int argc, char *argv[]) {

    // Input the matrix filename arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_file_A> <num_clusters>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file_A = argv[1];
    int numClusters = atoi(argv[2]);
    bool pprint_flag = false;

    // Check for the -pprint flag
    if (argc == 4 && strcmp(argv[3], "--pprint") == 0) {
        pprint_flag = true;
    }

    SparseMatrixCOO A, Omega;

    // Read from files
    if (readSparseMatrix(matrix_file_A, &A) != 0) {
        return EXIT_FAILURE;
    }

    int numNodes = A.M;  // Assuming A is a square matrix, numNodes is the number of rows/columns

    // Step 1: 
    // Construct the Omega matrix with random cluster assignments
    constructOmegaMatrix(&Omega, numNodes, numClusters);

    Omega.I[0] = 0;  Omega.J[0] = 1;  Omega.val[0] = 1.0;
    Omega.I[1] = 1;  Omega.J[1] = 1;  Omega.val[1] = 1.0;
    Omega.I[2] = 2;  Omega.J[2] = 2;  Omega.val[2] = 1.0;
    Omega.I[3] = 3;  Omega.J[3] = 0;  Omega.val[3] = 1.0;


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
    printf("\n<I> Total execution time: %f seconds\n", cpu_time_used);
    
    // Free memory
    freeSparseMatrix(&Omega);
    freeSparseMatrix(&M);

    return EXIT_SUCCESS;
}