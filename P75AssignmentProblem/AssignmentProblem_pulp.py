import pulp
import numpy as np
import random
from itertools import product
import pandas as pd

#ランダム設定 同じ条件で評価できるように
random.seed(1)
np.random.seed(1)

#仕事の数m、エージェント数n
m=10
n=5
#仕事の最大サイズ（調整用）
JOB_SIZE=10

#仕事jの資源要求量
a = np.random.randint(2,JOB_SIZE,size=(n,m))
#エージェントの利用可能資源量
b = np.random.randint(3,JOB_SIZE*2,size=n)
#コスト
c = np.random.randint(1,10,size=(n,m))

################################################################################
##### Pulpで解く
p = pulp.LpProblem("AssignmentProblem")
x = pulp.LpVariable.dict("x",indexs=(range(n),range(m)),lowBound=0,upBound=1,cat=pulp.LpBinary)

#目的関数定義
p += pulp.lpSum([x[(i,j)]*c[i,j] for i,j in product(range(n),range(m))])

#エージェントの利用可能資源量を超えない
for i in range(n):
    p += pulp.lpSum([x[(i,j)]*a[i,j] for j in range(m)]) <= b[i]

#全ての仕事をエージェントに割り振る
for j in range(m):
    p += pulp.lpSum([x[(i,j)] for i in range(n)]) == 1

p.solve()
#解が最適解であれば結果を表示
if p.status == 1:
    print("Optimization Result by Pulp")
    cols = []
    assigned_agents=[]
    for j in range(m):
        cols.append(f"JOB{j}")
        assigned_agents.append(int(sum(i*x[(i,j)].value() for i in range(n))))
    df = pd.DataFrame([assigned_agents],columns=cols,index=["result"])
    print(df)

    print(f"Value = {pulp.value(p.objective)}") 

elif p.status == -1:
    print("実行不能解")
    exit(0)