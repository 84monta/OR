from pulp import LpVariable,LpProblem
import random
from functools import wraps
import time
import gc

def time_checker(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        end = time.time()
        print(f"{end-start:.3f} sec")
        return result
    return wrapper


class Product:
    def __init__(self,value=None,weight=None):
        if value == None:
            value = random.randint(6,20) * 10
        if weight == None:
            weight = random.randint(2,10) * 10

        self.value  = value
        self.weight = weight

    def get_weight(self):
        return self.weight

    def get_value(self):
        return self.value

class Products:
    def __init__(self,num=10):
        self.p = []

        for i in range(num):
            self.p.append(Product())

        self.num = num

        average = sum(p.get_weight() for p in self.p) / 3
        self.limit_weight = int(average / 10)*10

    @time_checker
    def optimize_pulp(self):
        from pulp import LpVariable,LpProblem,LpMaximize,LpBinary,PULP_CBC_CMD

        p = LpProblem("knapsackP",LpMaximize)
        x = LpVariable.dict("x",indexs=(range(self.num)),lowBound=0,upBound=1,cat=LpBinary)

        #価値最大化
        p += sum(x[i]*self.p[i].get_value() for i in range(self.num))

        #重量を制限以内
        p += sum(x[i]*self.p[i].get_weight() for i in range(self.num)) <= self.limit_weight

        #解く
        solver=PULP_CBC_CMD(msg=0,threads=10)
        p.solve(solver)

        if p.status == 1:
            print("## Pulp Optimized Result")
            self.print([int(x[i].value()) for i in range(self.num)])
            ret = sum([x[i].value()*self.p[i].get_value() for i in range(self.num)])
            del x,p
            gc.collect()
            return ret
        else:
            print("## Pulp Optimizing Failed")
    
    #@time_checker
    def optimize_advantage(self,count=False,Best=0,balancer=1.0):
        from pyqubo import Array,Constraint,Placeholder,UnaryEncInteger #LogEncInteger
        from dwave.system import EmbeddingComposite,DWaveSampler

        x = Array.create("x",shape=(self.num),vartype="BINARY")
        #y = LogEncInteger("y",lower=0,upper=5)
        y = UnaryEncInteger("y",lower=0,upper=5)

        #価値が最大になるように（符号を反転させる）
        H1 = -sum([x[i]*self.p[i].get_value() for i in range(self.num)])

        #重さが目的の重量になるように
        H2 = Constraint((self.limit_weight -sum([x[i]*p_i.get_weight() for i,p_i in enumerate(self.p)]) - y*10)**2,"Const Weight")

        H = H1 + H2*Placeholder("balancer")
        model = H.compile()
        balance_dict = {"balancer":balancer}
        bqm = model.to_dimod_bqm(feed_dict=balance_dict)
        sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system1.1"))
        responses = sampler.sample(bqm,num_reads=1000)

        solutions = model.decode_dimod_response(responses,feed_dict=balance_dict)
        #Optuna用 バランス調査
        if count == True:
            counter = 0
            for idx,sol in enumerate(solutions):
                const_str = sol[1]
                val = sum(int(sol[0]['x'][i])*self.p[i].get_value()  for i in range(self.num))
                #重量が制限以下、かつ価値が最適解の9割以上をカウント
                if len(const_str) == 0 and val > Best*0.9:
                    counter += responses.record[idx][2]

            del H1,H2,H,model,bqm,responses,solutions
            gc.collect()

            return counter

        if len(solutions[0][1]) == 0:
            print(f" y= { sum(2**i*y_i for i,y_i in enumerate(solutions[0][0]['y']))}")
            print("## Advantage Solver Optimized Result")
            self.print([int(solutions[0][0]['x'][i]) for i in range(self.num)])
        else:
            print("## Advantage Solver Optimizing Failed")

    @time_checker
    def optimize_leap(self):
        from pyqubo import Array,Constraint,Placeholder,UnaryEncInteger #LogEncInteger
        from dwave.system import LeapHybridSampler

        x = Array.create("x",shape=(self.num),vartype="BINARY")
        #y = LogEncInteger("y",lower=0,upper=5)
        y = UnaryEncInteger("y",lower=0,upper=5)

        #価値が最大になるように（符号を反転させる）
        H1 = -sum([x[i]*self.p[i].get_value() for i in range(self.num)])

        #重さが目的の重量になるように
        H2 = Constraint((self.limit_weight -sum([x[i]*p_i.get_weight() for i,p_i in enumerate(self.p)]) - y*10)**2,"Const Weight")

        H = H1 + H2*Placeholder("balancer")
        model = H.compile()
        balance_dict = {"balancer":1.0}
        bqm = model.to_dimod_bqm(feed_dict=balance_dict)
        sampler = LeapHybridSampler()
        responses = sampler.sample(bqm,time_limit=10)

        solutions = model.decode_dimod_response(responses,feed_dict=balance_dict)

        if len(solutions[0][1]) == 0:
            print(f" y= { sum(2**i*y_i for i,y_i in enumerate(solutions[0][0]['y']))}")
            print("## LeapHybridSolver Optimized Result")
            self.print([int(solutions[0][0]['x'][i]) for i in range(self.num)])
        else:
            print("## LeapHybridSolver Optimizing Failed")

    def optimize_balancer(self):
        import optuna

        best_val = self.optimize_pulp()

        def objective(trial):
            balancer = trial.suggest_uniform('balancer',0.0,100.0)
            score = 0
            for i in range(10):
                score += self.optimize_advantage(count=True,Best=best_val,balancer=balancer)
            return score

        study_name = f"knapsack_size{self.num}"
        study = optuna.create_study(study_name=study_name,storage='sqlite:///optuna_study.db', load_if_exists=True,direction='maximize')

        study.optimize(objective, n_trials=100)

    def print(self,answer = None):
        sum_flag = True
        if answer == None:
            answer = [1]*self.num
            sum_flag = False

        print("-"*(9*sum(answer)+12))
        print("|          | ",end="")
        print(" | ".join(f"prod{i:02}" for i in range(self.num) if answer[i] == 1),end="")
        print(" |")
        print("-"*(9*sum(answer)+12))
        print("|  Weight  | ",end="")
        print(" | ".join(f"  {self.p[i].get_weight()} ".rjust(6, ' ') for i in range(self.num) if answer[i] == 1),end="")
        print(" |")
        print("-"*(9*sum(answer)+12))
        print("|  Value   | ",end="")
        print(" | ".join(f"  {self.p[i].get_value()} ".rjust(6, ' ') for i in range(self.num) if answer[i] == 1),end="")
        print(" |")
        print("-"*(9*sum(answer)+12))

        if sum_flag:
            print(f" Total Weight : {sum([self.p[i].get_weight() for i in range(self.num) if answer[i] == 1])} ,",end="")
            print(f" Total Value  : {sum([self.p[i].get_value() for i in range(self.num) if answer[i] == 1])}")

if __name__ == "__main__":
    #Range
    MAX_SIZE = 30
    #Benchmark Pulp
    # for i in range(5,31):
    #     random.seed(1)
    #     products = Products(num=i)
    #     #print(f"Weight Limit:{products.limit_weight}")
    #     #products.optimize_pulp()
    #     products.optimize_advantage()

    for i in range(10,31,5):
        random.seed(1)
        products = Products(i)
        products.optimize_balancer()

        del products
        gc.collect()

    #products.print()
    #print(f"Weight Limit:{products.limit_weight}")
    #products.optimize_pulp()
    #products.optimize_leap()
