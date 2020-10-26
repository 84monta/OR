from pulp import LpVariable,LpProblem,LpStatus
import networkx as nx

G = nx.DiGraph()
G.add_node("v1",b=10)
G.add_node("v2",b=0)
G.add_node("v3",b=0)
G.add_node("v4",b=0)
G.add_node("v5",b=-10)
G.add_edge("v1","v3",u=6,c=3)
G.add_edge("v1","v2",u=5,c=7)
G.add_edge("v2","v3",u=3,c=2)
G.add_edge("v2","v4",u=3,c=4)
G.add_edge("v3","v4",u=5,c=8)
G.add_edge("v3","v5",u=9,c=5)
G.add_edge("v4","v5",u=3,c=6)

#問題、変数定義
p = LpProblem("MCFP")
x = {}
for v in G.edges(data=True):
    x[f"x_{v[0]}_{v[1]}"]= LpVariable(f"x_{v[0]}_{v[1]}",lowBound=0,upBound=v[2]['u'],cat="Integer")

#目的関数定義(コスト最小化)
H = 0
for e in G.in_edges(data=True):
    H += x[f"x_{e[0]}_{e[1]}"] * e[2]["c"]
p += H    

#制約式
#各エッジの流量は容量以下
for e in G.edges(data=True):
   p += x[f"x_{e[0]}_{e[1]}"] <= e[2]['u']

#各ノードでの流出流入量合計が供給(b)と一致する
for v in G.nodes(data=True):
    tmp = 0
    b = v[1]
    #流入量
    for e in G.in_edges(v[0],data=True):
        tmp += x[f"x_{e[0]}_{e[1]}"] 
    #流出量
    for e in G.out_edges(v[0],data=True):
        tmp -= x[f"x_{e[0]}_{e[1]}"] 
    p += tmp + b == 0

stat = p.solve()
print(LpStatus[stat])

for v in G.edges(data=True):
   print(f"x_{v[0]}_{v[1]} =",x[f"x_{v[0]}_{v[1]}"].value())
