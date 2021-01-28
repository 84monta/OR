# From https://myenigma.hatenablog.com/entry/2020/12/23/201644

import itertools
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

score = np.random.rand(1000, 1000)
start = time.time()
row_ind, col_ind = linear_sum_assignment(score, maximize=True)
max_score = score[row_ind, col_ind].sum()
print(f"calc_time:{time.time() - start:.4f}[sec]")
print(f"{max_score}")

# brute force
start = time.time()
max_score = 0.0
max_col_ind = None
for v in itertools.permutations(range(score.shape[1]), score.shape[0]):
    sum_score = score[range(len(v)), v].sum()
    if max_score < sum_score:
        max_score = sum_score
        max_col_ind = v
print(f"calc_time:{(time.time() - start):.4f}[sec]")
print(f"{max_score}")