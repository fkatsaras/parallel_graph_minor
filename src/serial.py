class SparseMatrixCOO:

    def __init__(self, rows, cols, values, shape) -> None:
        """
        Initialize a sparse matrix in COO format.
        
        :param rows: List of row indices of non-zero elements
        :param cols: List of column indices of non-zero elements
        :param values: List of values of non-zero elements
        :param shape: Tuple representing the shape of the matrix (num_rows, num_cols)
        """
        self.rows = rows
        self.cols = cols
        self.values = values
        self.shape = shape

    def __str__(self) -> str:
        """
        String representation of the sparse matrix.
        """
        return f"SparseMatrixCOO(shape={self.shape}, rows={self.rows}, cols={self.cols}, values={self.values})"
    
    def multiply(self, other):
        """
        Multiply two sparse matrices in COO format.
        
        :param other: Another SparseMatrixCOO instance
        :return: Resultant SparseMatrixCOO instance after multiplication
        """

        # 1) Initialization
        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix dimensionsfor multiplication")
        
        result_dict = {}

        # 2) Multiplication process
        ### Iterate over all non-zero elements of A
        for a_i, a_j, a_val in zip(self.rows, self.cols, self.values):

            for b_j, b_k, b_val in zip(other.rows, other.cols, other.values):
                # Find matching elements in B (where row index equals A's column index)
                if a_j == b_j:
                    if (a_i, b_k) not in result_dict:
                        result_dict[(a_i, b_k)] = 0; # Initialize new result entry if it isn't already in the dict
                    result_dict[(a_i, b_k)] += a_val * b_val; # Multiply and add to result
        
        # 3) Construct result matrix
        result_rows, result_cols, result_values  = [], [], []
        for (i, j), value in result_dict.items():
            result_rows.append(i);
            result_cols.append(j);
            result_values.append(value);

        result_shape = (self.shape[0], other.shape[1]);
        return SparseMatrixCOO(result_rows, result_cols, result_values, result_shape);


# Exaple usage
A = SparseMatrixCOO([0,0,1], [0, 2, 1], [1, 2, 3], (2, 3))
B = SparseMatrixCOO([0, 1, 2], [0, 1, 2], [4, 5, 6], (3, 3))

C = A.multiply(B)
print(C)