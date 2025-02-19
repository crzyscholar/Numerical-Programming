import random
import numpy as np

def first_norm(matrix1, matrix2):
    norm = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            norm += abs(matrix1[i][j] - matrix2[i][j])
    return norm

def second_norm(matrix1, matrix2):
    diff = np.array(matrix1) - np.array(matrix2)
    norm = np.linalg.norm(diff, ord=2)
    return norm

def frobenius_norm(matrix1, matrix2):
    norm = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            norm += (matrix1[i][j] - matrix2[i][j]) ** 2
    return np.sqrt(norm)


r = 2  # rows
c = 4  # columns

matrix = [] 
print("Enter the entries row-wise (space-separated):")

# Getting matrix input
for i in range(r):
    row = list(map(int, input().split()))
    if len(row) != c:
        print(f"Error: You must enter exactly {c} values per row.")
        exit(1)
    matrix.append(row)


# for i in range(r):
#     for j in range(c):
#         print(matrix[i][j], end=" ")
#     print()



generated_matrices = []
for k in range(10):
    random_matrix = [[random.randint(0, 9) for k in range(c)] for k in range(r)]
    generated_matrices.append(random_matrix)


first_norm_distances = []
second_norm_distances = []
frobenius_norm_distances = []


for gen_matrix in generated_matrices:
    first_norm_distances.append(first_norm(matrix, gen_matrix))
    second_norm_distances.append(second_norm(matrix, gen_matrix))
    frobenius_norm_distances.append(frobenius_norm(matrix, gen_matrix))


min_first_norm_index = first_norm_distances.index(min(first_norm_distances))
max_first_norm_index = first_norm_distances.index(max(first_norm_distances))

min_second_norm_index = second_norm_distances.index(min(second_norm_distances))
max_second_norm_index = second_norm_distances.index(max(second_norm_distances))

min_frobenius_norm_index = frobenius_norm_distances.index(min(frobenius_norm_distances))
max_frobenius_norm_index = frobenius_norm_distances.index(max(frobenius_norm_distances))




def display_closest_and_farthest(norm_name, closest_idx, farthest_idx):
    print(f"\nUsing {norm_name}:")
    print("Closest Matrix:")
    for row in generated_matrices[closest_idx]:
        print(row)
    print("Farthest Matrix:")
    for row in generated_matrices[farthest_idx]:
        print(row)


print("\nInput Matrix:")
for row in matrix:
    print(row)


display_closest_and_farthest("First Norm", min_first_norm_index, max_first_norm_index)
display_closest_and_farthest("Second Norm", min_second_norm_index, max_second_norm_index)
display_closest_and_farthest("Frobenius Norm", min_frobenius_norm_index, max_frobenius_norm_index)


#pirveli norma aritvaliswinebs or ganzomilebas da 
#prosta pirdapir aklebs matricas ori ganzomileba aqvs
#frobeniuss imitom jobia rom frobeniusi prosta yvelas ert wonas 
#anichebs da ise itvlis imis miuxedavad rom matrica sheidzleba sadme 
#iyos gadaxrili ufro metad vidre meore mxares da frobeniusi prosta yvelas 
#akvadratebs da itvlis magis gautvaliswineblad