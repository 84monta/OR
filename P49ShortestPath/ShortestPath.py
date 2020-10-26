from matplotlib import pyplot as plt
import networkx  as nx
from pulp import LpProblem,LpMinimize,LpStatus,LpVariable,LpBinary
from random import randrange,sample,seed
import logging

class problem:
    def __init__(self,n=10,p=1,random_seed=1):
        #self.G = nx.directed_configuration_model()
        seed(random_seed)
        self.n = n
        self.nodes = list(range(n))
        self.start,self.goal = sample(self.nodes,2)
        self.G = nx.DiGraph()
        #empty_graph(directed=)
        for i in self.nodes:
            self.G.add_node(i)
        
        #ゴールまでのパスを生成する
        path = self.nodes.copy()
        path.remove(self.start)
        path.remove(self.goal)
        path = [self.start] + sample(path,int(n/2) + randrange(0,int(n/4))) + [self.goal]
        for i in range(len(path)-1):
            self.G.add_edge(path[i],path[i+1])

        #独立しているNodeをなくす
        independent_node = list(set(self.nodes) - set(path))
        for u in independent_node:
            v = sample(path,1)[0]
            self.G.add_edge(u,v)
        
        #ランダムなEdgeを追加する
        for i in range(int(n*p)):
            u,v = sample(self.nodes,2)
            if not (self.G.has_edge(u,v) or self.G.has_edge(v,u)):
                self.G.add_edge(u,v)

        #self.pos = nx.layout.spring_layout(self.G)
        self.pos = nx.layout.kamada_kawai_layout(self.G)
        self.edges = list(self.G.edges)
        self.edges_num = len(self.edges)

        for (u,v) in self.edges:
            self.G.edges[u,v]['w'] = randrange(3,10)
        
        logging.debug(f"Nodes {self.nodes}")
        logging.debug(f"Edges {self.edges}")
        logging.debug(f"Designed path = {path}")
        

    def solve_pulp(self,start=0,end=4):
        #最小化問題
        p = LpProblem("ShortestPath",sense=LpMinimize)
        #変数
        x = LpVariable.dicts('x',range(len(self.edges)),0,1,cat = LpBinary)

        #目的関数
        p += sum([ x[i]*self.get_weight(i) for i in range(self.edges_num)])
        #p += sum([ x[i] for i in range(self.edges_num)])

        #制約式
        for i in self.nodes:
            const_out = 0
            tmp_edges = []
            for e in self.G.edges(i):
                j = e[1]
                const_out += x[self.get_node_index((i,j))]
                tmp_edges.append(self.get_node_index((i,j)))
            logging.debug(tmp_edges)
            
            const_in = 0
            tmp_edges = []
            for e in self.G.in_edges(i):
                k = e[0]
                const_in += x[self.get_node_index((k,i))]
                tmp_edges.append(self.get_node_index((k,i)))
            logging.debug(tmp_edges)

            print(i,self.G.edges(i),self.G.in_edges(i))

            #self.G.edges()
            if i == self.start:
                const_val = 1
            elif i == self.goal:
                const_val = -1
            else:
                const_val = 0

            p += const_out - const_in == const_val
        
        self.status = p.solve()
        if self.status == 1:
            self.answer = [self.edges[i] for i in range(self.edges_num) if x[i].value() == 1]
            logging.debug(f"ans {self.answer}")
        else:
            logging.error("Solve Failed")

        print(self.answer)

    def show(self):
        plt.figure(figsize=(25,25),dpi=100)
        nx.draw_networkx(self.G,self.pos)
        nx.draw_networkx_nodes(self.G,self.pos,nodelist=[self.start,self.goal],node_color='y')
        print(self.status)
        if self.status == 1:
            nx.draw_networkx_edges(self.G,self.pos,edgelist=self.answer,edge_color='r')
            edge_labels = {}
            for u,v in self.edges:
                edge_labels[(u,v)] = self.G.edges[u,v]['w']
            nx.draw_networkx_edge_labels(self.G,self.pos,edge_labels=edge_labels)
        plt.show()


    def get_weight(self,i):
        (u,v) = self.edges[i]
        return(self.G.edges[u,v]['w'])

    def get_node_index(self,e):
        return(self.edges.index(e))
         
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    p = problem(n=100,p=1,random_seed=4)
    p.solve_pulp()

    p.show()
