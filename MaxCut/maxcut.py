from dwave.system.samplers.dwave_sampler import DWaveSampler
from pyqubo import Array,Sum,Constraint
from dwave.system import FixedEmbeddingComposite,EmbeddingComposite
from pulp import LpProblem,LpMaximize,lpSum,LpVariable,LpBinary
import networkx as nx
from matplotlib import pyplot as plt
import time
import argparse
import random
import picos as pic
import numpy as np

class MyTimer():
    '''
    時刻計測用 のクラス
    '''
    def __init__(self):
        self.set_timer()
            
    def set_timer(self):
        self.MY_TIMER = time.time()

    def get_timer(self):
        current_time = time.time()
        ret_time = current_time - self.MY_TIMER
        self.MY_TIMER = current_time
        return(ret_time)

def get_args():
    parser = argparse.ArgumentParser(description='This is sample argparse script')
    parser.add_argument('-n', '--node', default=20, type=int, help='This is number of nodes')
    parser.add_argument('-p', '--percentage', default=0.3, type=float, help='This is Probability for edge creation')
    parser.add_argument('-s', '--seed', default=0,type=int, help='This is seeds of random')

    return parser.parse_args()

class Problem():
    def __init__(self,n,p,seed):
        self.p = p
        self.n = n
        self.seed = seed
        self.G = nx.fast_gnp_random_graph(n,p,seed=seed)

        for v in self.G.nodes:
            if len(self.G.edges(v)) == 0:
                self.G.add_edge(v,random.randint(0,n))
        
        #重みづけ
        for (u,v) in self.G.edges:
            self.G.edges[u,v]['w'] = random.randrange(0,30,1)


        self.pos = nx.layout.spring_layout(self.G)

    
    def solve_pulp(self):
        P = LpProblem("maxcut",LpMaximize)
        X = LpVariable.dicts('X',range(self.n),0,1,LpBinary)

        #目的関数
        tmp_obj = 0
        # for v in self.G.nodes:
        #     for e in self.G.edges(v):

    def solve_picos(self):
        #http://www.orsj.or.jp/archive2/or63-12/or63_12_755.pdf
        #とけてるのかどうかわからん
        p = pic.Problem()
        X = p.add_variable('X',(self.n,self.n),'symmetric')
        gL = nx.laplacian_matrix(self.G,weight='w',nodelist=self.G.nodes)
        gL =gL.toarray().astype(np.double)
        L = pic.new_param('L',1/4 * gL)
        p.add_constraint(pic.diag_vect(X)==1)
        p.add_constraint(X>>0)
        p.set_objective('max',L|X)
        p.solve()
        print('bound from the SDP relaxation: {0}'.format(p.obj_value()))
        #print(X.value)
    
    def solve_dwave(self):
        X = Array.create('X',shape=(self.n),vartype='SPIN')

        H0 = 0
        for (u,v) in self.G.edges:
            H0 -= (1 - X[u]*X[v])*self.G.edges[u,v]['w']

        model = H0.compile()
        bqm = model.to_dimod_bqm()

        response = EmbeddingComposite(DWaveSampler()).sample(bqm,num_reads=100)
    
        result = model.decode_dimod_response(response,topk=1)
        optimized_v = result[0][0]['X']

        self.v_group = []
        for i in range(self.n):
            self.v_group.append(optimized_v[i])

        self.cutted = []
        for (u,v) in self.G.edges:
            if optimized_v[u] != optimized_v[v]:
                self.cutted.append([u,v])

    def show(self):
        plt.figure(figsize=(12,12),dpi=100)
        nx.draw_networkx(self.G,pos=self.pos)
        if len(self.cutted)>0:
            nx.draw_networkx_edges(self.G,self.pos,self.cutted,width=2,edge_color='r')
        nx.draw_networkx_edge_labels(self.G,self.pos,edge_label=[self.G.edges[u,v]['w'] for (u,v) in self.G.edges])
        plt.show()



if __name__ == '__main__':
    args = get_args()

    p = Problem(n=args.node,p=args.percentage,seed=args.seed)
    #p.solve_pulp()
    #p.solve_picos()
    p.solve_dwave()
    p.show()