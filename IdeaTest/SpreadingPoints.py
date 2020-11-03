from pulp import LpVariable,LpProblem,LpBinary,LpContinuous,LpMaximize,LpStatus,value,PULP_CBC_CMD
from math import sqrt
from itertools import product

######定数定義
#格子の数
SIZE = 10
#ばらまくPointの数
N = 6

#2点間の距離を算出する
def dist(p1,p2):
    return(sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/SIZE)

######変数定義
#どこにPointを置くか
x = LpVariable.dict(name="x",indexs=(range(SIZE),range(SIZE)),lowBound=0,upBound=1,cat=LpBinary)
#今回のキモ 多目的のための変数(どんなに良くても対角の√2以下)
z = LpVariable("z",lowBound=0,upBound=1.5,cat=LpContinuous)

#問題定義
p = LpProblem(name="SpreadingPointsProblem",sense=LpMaximize)

#zの最大化問題として定義
p += z

#N個のPointが配置される
tmp_sub = 0
for i,j in product(range(SIZE),range(SIZE)):
    tmp_sub += x[(i,j)]
p += tmp_sub == N

#確認した2点に同時に点が配置されているとしたばあい、その距離はz以上
BIGM = 100.0
for i,j in product(range(SIZE),range(SIZE)):
    for k,l in product(range(SIZE),range(SIZE)):
        if i==k and j==l:
            continue
        #x[(i,j)] x[(k,l)]が両方1のときのみ意味ある不等式となる。
        p += z <= (1-x[(i,j)])*BIGM + (1-x[(k,l)])*BIGM + dist((i,j),(k,l))

solver = PULP_CBC_CMD(msg=0,mip=True,threads=15)
p.solve(solver)
print(LpStatus[p.status])
print(f"Best Distance = {value(p.objective)}")

#結果出力
for j in range(SIZE):
    print(",".join(str(int(x[(j,i)].value())) for i in range(SIZE)))