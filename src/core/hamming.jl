"""
Hamming Distance Computation Functions

This module provides functions to compute Hamming distances between matrices.
The Hamming distance measures the number of positions at which corresponding elements differ.

Author: HICMA Team
"""

"""
    hamming_binary(X, Y=nothing)

Compute binary Hamming distance matrix for binary matrices.

# Arguments
- `X`: Binary input matrix (m × n) where elements are 0 or 1
- `Y`: Optional binary input matrix (m × n). If nothing, computes self-distance

# Returns
- `D`: Hamming distance matrix (n × n)

# Details
For self-distance (Y = nothing):
- D[i,j] = number of positions where X[:,i] and X[:,j] differ
- Uses efficient matrix operations: D = (1-X)' * X + X' * (1-X)

For cross-distance (Y provided):
- D[i,j] = number of positions where X[:,i] and Y[:,j] differ
- Uses: D = (1-X)' * Y + X' * (1-Y)
"""
function hamming_binary(X, Y=nothing)
    m, n = size(X)
    
    if Y == nothing
        # Self-distance computation: D[i,j] = Hamming distance between columns i and j
        # Formula: D = (1-X)' * X + X' * (1-X)
        # This counts positions where one column has 1 and the other has 0
        D = (ones(m, n) - X)' * X
        D = D + D'  # Make symmetric since D[i,j] = D[j,i]
    else
        # Cross-distance computation: D[i,j] = Hamming distance between X[:,i] and Y[:,j]
        # Formula: D = (1-X)' * Y + X' * (1-Y)
        # This counts positions where X has 1 and Y has 0, plus X has 0 and Y has 1
        D = (ones(m, n) - X)' * Y + X' * (ones(n, m) - Y)
    end
    
    return D
end

"""
    hamming(X, Y=nothing)

Compute general Hamming distance matrix for matrices with arbitrary discrete values.

# Arguments
- `X`: Input matrix (m × n) with discrete values
- `Y`: Optional input matrix (m × n). If nothing, computes self-distance

# Returns
- `H`: Hamming distance matrix (n × n) normalized by 2

# Details
This function generalizes binary Hamming distance to arbitrary discrete values by:
1. Converting each unique value to a binary indicator matrix
2. Computing binary Hamming distance for each indicator
3. Summing all binary distances
4. Dividing by 2 to avoid double-counting

The result gives the total number of positions where corresponding elements differ.
"""
function hamming(X, Y=nothing)
    m, n = size(X)
    
    if Y == nothing
        # Self-distance: find unique values in X and compute binary Hamming for each
        uniqs = unique(X)
        H = hamming_binary(X .== uniqs[1])  # Start with first unique value
        
        # Add binary Hamming distances for all other unique values
        for uni in uniqs[2:end]
            H += hamming_binary(X .== uni)
        end
    else
        # Cross-distance: find union of unique values from both X and Y
        uniqs = union(X, Y)
        H = hamming_binary(X .== uniqs[1], Y .== uniqs[1])  # Start with first unique value
        
        # Add binary Hamming distances for all other unique values
        for uni in uniqs[2:end]
            H += hamming_binary(X .== uni, Y .== uni)
        end
    end
    
    # Divide by 2 to avoid double-counting positions
    return H / 2
end