import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import random
import time
import logging
import pandas as pd
from numba import jit

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
        self.E = list(self.G.edges())
        self.V = list(self.G.nodes())

        #選択されたEdge 
        self.x = []
        #Prim法で用いる、選択済みのノード
        Vnew = []
        #各ループで新たに選択するノード
        selected_node = 0
        Vnew.append(selected_node)

        t0 = time.time()
        #Edge 番号、重みのリストを作成し、重みが小さい順でsort
        while Vnew != self.V:
            
            #最小WeightのEdgeを探す。
            min_weight = 11
            selected_edge = -1
            for v in Vnew:
                tmp_edges = self.G.edges(v)
                for tmp_edge in tmp_edges:
                    u = tmp_edge[0]
                    v = tmp_edge[1]
                    if u > v:
                        u,v = v,u
                    #両方のノードがVnewに含まれていたら対象外
                    if u in Vnew and v in Vnew:
                        continue
                    tmp_weight = self.G.edges[u,v]['weight'] 
                    if tmp_weight < min_weight:
                        min_weight = tmp_weight
                        if u in Vnew:
                            selected_node = v
                        else:
                            selected_node = u
                        selected_edge = self.E.index((u,v))
            self.x.append(selected_edge)
            #ノードを追加してソート
            Vnew.append(selected_node)
            Vnew.sort()

        t1 = time.time()
        self.time_solve = round(t1 - t0,2)
        #logging.debug(f"Prim method : {self.time_solve} sec") 
        logging.debug("Prim method : %f sec",self.time_solve) 
        self.stat = 1
            

    def PrintTime(self):
        if self.stat == 1:
            print(f"Solve {self.time_solve} " )
        else:
            print(f"Not Solved" )

    def ShowAnswer(self):
        if self.stat != 1:
            print("Problem is not solved")
            return
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(self.G,self.pos)
        tree_edge = [self.E[idx] for idx in self.x]
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
        return(self.time_solve)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.CRITICAL)

    p = []
    spent_times = []
    psizes = []
    for i in range(100,300,10):
        new_p = ProblemGraph(size=i)
        new_p.Solve()
        new_p.PrintTime()
        #new_p.ShowAnswer()
        p.append(new_p)
        spent_times.append(new_p.GetTimes())
        psizes.append(i)
    dataset = pd.DataFrame(spent_times,columns=['Solve'],index=psizes).T

    fig, ax = plt.subplots(figsize=(10, 8))
    dataset.T.plot(kind='bar', stacked=True, ax=ax)
    plt.show()