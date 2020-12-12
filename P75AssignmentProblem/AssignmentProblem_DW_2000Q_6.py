from pyqubo import Array,Sum,Constraint,Placeholder,OneHotEncInteger,UnaryEncInteger,LogEncInteger
from dwave.system import EmbeddingComposite,DWaveSampler,LeapHybridSampler
import pulp
import numpy as np
import random
from itertools import product
import pandas as pd
import optuna

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
### D-Waveで解く

##パラメータ探索用
def objective(trial):
    b1 = trial.suggest_uniform('b1',0.0,20)
    b2 = trial.suggest_uniform('b2',0.0,20)
    b3 = trial.suggest_uniform('b3',0.0,20)

    def_dict = {"balancer1":b1,"balancer2":b2,"balancer3":b3,"IntChain":1.0}
    bqm = model.to_dimod_bqm(feed_dict=def_dict)

    sampler = EmbeddingComposite(DWaveSampler(sampler="DW_2000Q_6"))
    responses = sampler.sample(bqm,num_reads=5000)

    solutions = model.decode_dimod_response(responses,feed_dict=def_dict)

    cnt = 0

    for idx,sol in enumerate(solutions):
        if len(sol[1]) < 10:
            cnt += responses.record[idx][2]

    return cnt

#定義上書き！
x = Array.create('x',shape=(n,m),vartype="BINARY")
#不等号表現用の補助スピン
y = []
for i in range(n):
    #y.append(OneHotEncInteger(f"y{i}",lower=0,upper=JOB_SIZE*2,strength=Placeholder("IntChain")))
    y.append(LogEncInteger(f"y{i}",lower=0,upper=JOB_SIZE*2))

#目的関数定義相当
H1 = Sum(0,m,lambda j: Sum(0,n,lambda i: x[i][j]*c[i,j]))

#エージェントの利用可能資源量を超えない
H2 = Sum(0,n,lambda i: Constraint(Sum(0,m,lambda j: x[(i,j)]*a[i,j] +y[i] - b[i])**2,f"Agent Resource {i}"))
#H2 = Sum(0,n,lambda i: Sum(0,m,lambda j: x[(i,j)]*a[i,j] - b[i])**2)

#全ての仕事をエージェントに割り振る
H3 = Sum(0,m,lambda j: Constraint(Sum(0,n,lambda i: x[i][j]-1)**2 ,f"Job{j}"))

H = Placeholder("balancer1")*H1 + Placeholder("balancer2")*H2 + Placeholder("balancer3")*H3

model = H.compile()

#パラメータ探索
study = optuna.create_study(storage='sqlite:///example.db',study_name=f"m{m}n{n}_DW2000Q",load_if_exists=True)
study.optimize(objective, n_trials=200)


b1 = study.best_params["b1"]
b2 = study.best_params["b2"]
b3 = study.best_params["b3"]

def_dict = {"balancer1":b1,"balancer2":b2,"balancer3":b3,"IntChain":10.0}
bqm = model.to_dimod_bqm(feed_dict=def_dict)

sampler = EmbeddingComposite(DWaveSampler(sampler="DW_2000Q_6"))
responses = sampler.sample(bqm,num_reads=1000)

#sampler = LeapHybridSampler()
#responses = sampler.sample(bqm,time_limit=60)

solutions = model.decode_dimod_response(responses,feed_dict=def_dict)

cnt = 0

for sol in solutions:
    if len(sol[1]) < 10:
        weight_const_flag = False
        for i in range(n):
            if sum(sol[0]['x'][i][j] * a[i,j] for j in range(m)) > b[i]:
                weight_const_flag = True

        #エージェントのリソースを超えている場合は答え候補を無視 
        if (weight_const_flag) :
            continue

        print(sol[0])
        print(sol[1])
        print(sol[2])
        cnt += 1

print("No penalty answer count is",cnt)