# Get matrix dimensions from the user
rows_a = int(input("Enter the number of rows for the first matrix: "))
cols_a = int(input("Enter the number of columns for the first matrix: "))

rows_b = cols_a  # To multiply, the second matrix's rows must equal the first matrix's columns
cols_b = int(input("Enter the number of columns for the second matrix: "))

# File to write the matrices
mat_A_file = "../testA.mtx.sty"
mat_B_file = "../testB.mtx.sty"

header = """%%MatrixMarket matrix coordinate real symmetric
%-------------------------------------------------------------------------------
% UF Sparse Matrix Collection, Tim Davis
% http://www.cise.ufl.edu/research/sparse/matrices/Pajek/GD97_b
% name: Pajek/GD97_b
% [Pajek network: Graph Drawing contest 1997]
% id: 1493
% date: 1997
% author: Graph Drawing Contest
% ed: V. Batagelj
% fields: name title A id kind notes aux date author ed
% aux: nodename coord
% kind: undirected weighted graph
%-------------------------------------------------------------------------------
% notes:
% ------------------------------------------------------------------------------
% Pajek network converted to sparse adjacency matrix for inclusion in UF sparse 
% matrix collection, Tim Davis.  For Pajek datasets, See V. Batagelj & A. Mrvar,
% http://vlado.fmf.uni-lj.si/pub/networks/data/.                                
% ------------------------------------------------------------------------------
% Regarding conversion for UF sparse matrix collection: in the original data    
% every edge appears exactly twice, with the same edge weight.  It could be a   
% multigraph, but it looks more like a graph.  The duplicate edges are removed  
% in this version.  You can always add them back in yourself; just look at 2*A. 
% ------------------------------------------------------------------------------
% The original problem had 3D xyz coordinates, but all values of z were equal   
% to 0.5, and have been removed.  This graph has 2D coordinates.                
%-------------------------------------------------------------------------------
"""

# Write matrix A
with open(mat_A_file, 'w') as file:

    # Write description
    file.write(header)
    # Write dimensions
    file.write(f'{rows_a} {cols_a} {rows_a * cols_a}\n')

    # Write matrix entries
    for i in range(1, rows_a + 1):
        for j in range(1, cols_a + 1):
            file.write(f'{i} {j} 1\n')

# Write matrix B
with open(mat_B_file, 'w') as file:

    # Write description
    file.write(header)
    # Write dimensions
    file.write(f'{rows_b} {cols_b} {rows_b * cols_b}\n')

    # Write matrix entries
    for i in range(1, rows_b + 1):
        for j in range(1, cols_b + 1):
            file.write(f'{i} {j} 1\n')

print(f"Matrices have been written to {mat_A_file}, {mat_B_file}.")