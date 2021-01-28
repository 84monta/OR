from pulp import LpProblem,LpVariable,LpBinary,lpSum,value,CPLEX_CMD
import random
import numpy as np
from itertools import product

p = LpProblem("mwpmp")

#問題作成
random.seed(10)
PSIZE = 30

w = np.random.uniform(0,1,size=(PSIZE,PSIZE))

x = LpVariable.dict(name="x",indexs=(range(PSIZE),range(PSIZE)),cat=LpBinary)

p += lpSum(x[i,j]*w[i,j] for i,j in product(range(PSIZE),range(PSIZE)))

for i in range(PSIZE):
    p += lpSum(x[i,j] for j in range(PSIZE)) == 1
    p += lpSum(x[j,i] for j in range(PSIZE)) == 1

#solver=CPLEX_CMD()
#p.solve(solver)
p.solve()

for i,j in product(range(PSIZE),range(PSIZE)):
    if value(x[i,j]) == 1:
        print(f"Matching ({i},{j})")