import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from pulp import LpProblem,LpVariable,LpMinimize,lpSum,LpStatus,PULP_CBC_CMD
import random
import time
import logging
import pandas as pd

class MyTimer():
    '''
    時刻計測用 のクラス
    '''
    #MY_TIMER = 0

    def set_timer(self):
        self.MY_TIMER = time.time()

    def get_timer(self):
        return(time.time() - self.MY_TIMER)

class ProblemGraph():
    def __init__(self,size=10,p=0.4,seed=0):
        #ランダムなGraphを作成
        self.G = nx.fast_gnp_random_graph(size,p,seed=seed)
        #もし、どのedgeにも属していないNodeがあれば、適当なNodeと連結する。
        node_list = list(self.G.nodes())
        for i in self.G.nodes:
            if len(self.G.edges(i)) == 0:
                tmp_node_list = node_list.copy()
                tmp_node_list.remove(i)
                v2 = random.choice(tmp_node_list)
                self.G.add_edge(i,v2)

        for (u,v) in self.G.edges():
            self.G.edges[u,v]['weight'] = random.randint(1,10)
        self.pos = nx.spring_layout(self.G)

    def Solve(self):
        mytimer = MyTimer()
        #Edge 番号があるかどうかわからないので、エッジの集合を作成し、Indexを番号とする。
        self.edges = list(self.G.edges())

        #Pulpで変数定義。普通にi番目のエッジを選択したときに１となる変数X[i]を定義
        P = LpProblem("MinimumSpanningTree",LpMinimize)
        X = LpVariable.dicts('X',range(len(self.edges)),0,1,'Binary')
        mytimer.set_timer()
        #目的関数
        tmp_obj = 0
        for idx,e in enumerate(self.edges):
            tmp_obj +=  X[idx] * self.G.edges[e[0],e[1]]['weight']
        P += tmp_obj

        self.time_objective = round(mytimer.get_timer(),3)
        logging.debug("Creating objective was taken : %f sec",self.time_objective) 

        #制約式1. 全ての閉路よりも1以上少ないEdgeが選ばれる
        #有効グラフ用simple_cyclesを使う
        mytimer.set_timer()
        G2 = self.G.to_directed()
        cycles = nx.simple_cycles(G2)
        for cycle in cycles:
            tmp_constraints = 0
            if  len(cycle) == 2:
                continue
            #閉路のすべてのエッジ（の変数）を足す
            for idx in range(len(cycle)):
                v1 = cycle[idx]
                v2 = cycle[(idx+1)%len(cycle)]
                edge = (v1,v2) if v1 < v2 else (v2,v1)
                edge_id = self.edges.index(edge)
                tmp_constraints += X[edge_id]
            #合計が閉路の長さより―1 以上短い
            P += tmp_constraints <= len(cycle) -1

        self.time_t1= round(mytimer.get_timer() ,3)
        logging.debug("Creating Constraints 1 was taken : %f sec",self.time_t1) 

        #制約式2. エッジの数が頂点数-1と同値
        mytimer.set_timer()
        P += lpSum(X) == len(self.G.nodes()) - 1
        t3 = time.time()
        self.time_t2= round(mytimer.get_timer(),3)
        logging.debug("Creating Constraints 2 was taken : %f sec",self.time_t2) 

        #solver = PULP_CBC_CMD(msg=0,threads=12,mip=True,maxSeconds=1000)
        mytimer.set_timer()
        solver = PULP_CBC_CMD(msg=True,threads=12,mip=True,maxSeconds=1000)
        self.stat = P.solve(solver)
        self.time_solve= round(mytimer.get_timer(),3)
        logging.debug("Solving problem was taken : %f sec",self.time_solve) 
        logging.info("Solved status is %s ",LpStatus[self.stat]) 

        self.x = []
        if self.stat == 1:
            for i in range(len(self.edges)):
                self.x.append(int(X[i].value()))
        else:
            logging.critical("Failed")

    def PrintTime(self):
        if self.stat == 1:
            print(f"size {len(self.G.nodes)} " , end="")
            print(f"Ojbective {self.time_objective} " , end="")
            print(f"Const1 {self.time_t1} " , end="")
            print(f"Const2 {self.time_t2} " , end="")
            print(f"Solve {self.time_solve} " )
        else:
            print(f"Not Solved" )

    def ShowAnswer(self):
        if self.stat != 1:
            print("Problem is not solved")
            return
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(self.G,self.pos)
        tree_edge = [self.edges[idx] for idx,val in enumerate(self.x) if val == 1]
        nx.draw_networkx_edge_labels(self.G,self.pos, edge_labels={(u,v) : self.G.edges[u,v]['weight'] for u,v in self.G.edges()})
        nx.draw_networkx_edges(self.G,self.pos,with_labels=True,edgelist=tree_edge,width=3.0,edge_color='red')
        plt.ioff()
        plt.show()

    def ShowProblem(self):
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edge_labels(self.G,self.pos, edge_labels={(u,v) : self.G.edges[u,v]['weight'] for u,v in self.G.edges()})
        nx.draw_networkx(self.G,self.pos,with_labels=True)
        plt.ioff()
        #plt.ion()
        plt.show()

    def GetTimes(self):
        return([self.time_objective,self.time_t1,self.time_t2,self.time_solve])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.CRITICAL)

    p = []
    ##ノード数は15くらいが限界（閉路のリストアップに時間がかかる)
    spent_times = []
    psizes = []
    for i in range(5,18):
        new_p = ProblemGraph(size=i)
        new_p.Solve()
        new_p.PrintTime()
        #new_p.ShowAnswer()
        p.append(new_p)
        spent_times.append(new_p.GetTimes())
        psizes.append(i)
    dataset = pd.DataFrame(spent_times,columns=['Objective','Const1','Const2','Solve'],index=psizes).T

    fig, ax = plt.subplots(figsize=(10, 8))
    dataset.T.plot(kind='bar', stacked=True, ax=ax)
    plt.show()

    #p.PrintTime()
    #p.ShowAnswer()