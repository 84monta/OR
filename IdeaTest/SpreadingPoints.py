from pulp import LpVariable,LpProblem,LpBinary,LpContinuous,LpMaximize,COIN_CMD,LpStatus,value
from math import sqrt
from itertools import product

######定数定義
#格子の数
SIZE = 10
#ばらまくPointの数
N = 5

#2点間の距離を算出する
def dist(p1,p2):
    return(sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/SIZE)

######変数定義
#どこにPointを置くか
x = LpVariable.dict(name="x",indexs=(range(SIZE),range(SIZE)),lowBound=0,upBound=1,cat=LpBinary)
#Slack変数
y = LpVariable.dict(name="y",indexs=(range(SIZE),range(SIZE),range(SIZE),range(SIZE)),lowBound=0,upBound=1,cat=LpBinary)
#今回のキモ 多目的のための変数
z = LpVariable("z",lowBound=0,upBound=2,cat=LpContinuous)

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
        #tmp_sumは2つの点が両方1になるときのみ１となる(tmp_sumは-1,0,1をとりうるが、目的関数がzの最大化のため、0,1のいずれかをとる。)
        tmp_sum = x[(i,j)] + x[(k,l)] - y[(i,j,k,l)]
        #p += z <= tmp_sum*dist((i,j),(k,l)) + (tmp_sum -1)*BIGM
        p += z <= (1-x[(i,j)])*BIGM + (1-x[(k,l)])*BIGM + dist((i,j),(k,l))
        #p += z <= dist((i,j),(k,l))*x[(i,j)]
        #p += z <= dist((i,j),(k,l))*x[(k,l)]
        #p += z <= dist((i,j),(k,l))*tmp_sum + (1-x[(i,j)])*1000.0 + (1-x[(k,l)])*1000.0

solver = COIN_CMD(mip=True,threads=10)
p.solve()
print(LpStatus[p.status])
print(f"Best Distance = {value(p.objective)}")

#結果出力
for j in range(SIZE):
    print(",".join(str(int(x[(j,i)].value())) for i in range(SIZE)))