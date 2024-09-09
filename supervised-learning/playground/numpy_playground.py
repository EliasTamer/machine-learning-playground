import math, copy
import numpy as np

vector = np.array([1,2,3,4])
negated_vector = np.array([-1,-2,-3,-4])

# will print the dimensions of this vector
print(vector.shape)

# will print the data type of the vector elements
print(vector.dtype)

# will print the first element of the vector
print(vector[1])

# will slice the vector
print(vector[1:3])

# will print the sum of all elements inside the vector
print(np.sum(vector))

# will sum 2 vectors together
print(vector + negated_vector)

# will multiply the vector elements by 5
print(5 * vector)

# multiplies the values in two vectors element-wise and then sums the result
print(np.dot(vector, vector))


matrice = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ])

# matrices include a second index. The two indexes describe [row, column].
print(matrice[0][1])

# accessing the 2nd row, and slicing it aftewards (start:stop:step)
print(matrice[1, 2:7:1])

