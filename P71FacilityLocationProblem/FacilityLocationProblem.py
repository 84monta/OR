from pulp import LpProblem,lpSum,LpBinary,LpContinuous,LpVariable,LpMinimize
import random
import numpy as np
from itertools import product

random.seed(1)

#顧客の集合(顧客は主にjで表す)
D_SIZE = 30

#施設候補場所(施設は主にiで表す)
F_SIZE = 5

###問題設定
#施設建設にかかるコスト
f = np.random.randint(low=100,high=200,size=F_SIZE)
#各施設から顧客への輸送コスト
c = np.random.randint(low=10,high=100,size=(F_SIZE,D_SIZE))

p = LpProblem(name="FLP",sense=LpMinimize)

#施設iから顧客jに運ぶ製品の割合（どれだけ需要を満たすかの割合）
x = LpVariable.dict('x',indexs=(range(F_SIZE),range(D_SIZE)),lowBound=0,upBound=1,cat=LpContinuous)

#施設iを開設するか(1)しないか(0)を表す変数y
y = LpVariable.dict('y',indexs=(range(F_SIZE)),lowBound=0,upBound=1,cat=LpBinary)

#目的関数定義
p += lpSum([f[i]*y[i] for i in range(F_SIZE)]) + lpSum([c[(i,j)]*x[(i,j)] for i,j in product(range(F_SIZE),range(D_SIZE))])

#施設のありなしによって、製品をその施設から出荷できるかどうか
for i,j in product(range(F_SIZE),range(D_SIZE)):
    p += x[(i,j)] <= y[i]

#割合の合計は1
for j in range(D_SIZE):
    p += lpSum([x[(i,j)] for i in range(F_SIZE)]) == 1.0

#解く
p.solve()

#最適化できたら解を出力
if p.status == 1:
    print(", ".join(f"y[{i}]={int(y[i].value())}" for i in range(F_SIZE)))
    for j in range(D_SIZE):
        print(", ".join(f"x[{i:02},{j:02}]={x[(i,j)].value()}" for i in range(F_SIZE)))