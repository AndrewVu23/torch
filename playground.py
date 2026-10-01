import numpy as np;
import matplotlib as mpl;

array1 = np.array([[['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
                    [['J', 'K', 'L'], ['M', 'N', 'O'], ['P', 'Q', 'R']],
                    [['S', 'T', 'U'], ['V', 'W', 'X'], ['W', 'Z', '']]])

array2 = np.array([[1, 2, 3, 4], 
                    [5, 6, 7 ,8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])

array3 = np.array([1, 2, 3 ,4])
array4 = np.array([[1], [2], [3], [4]])

print(np.mean(array3))
print(np.std(array4))
