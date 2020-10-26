from pulp import LpProblem,LpConstraint,LpVariable,LpBinary,LpInteger,LpStatus,PULP_CBC_CMD
from random import randint,seed

#Define Jobs
class job():
    def __init__(self,term,max_p):
        self.p = randint(1,max_p)
        self.r = randint(0,term-self.p*2)
        self.d = randint(self.r+self.p,term)

    def print(self):
        print(f"r={self.r},d={self.d},p={self.p}")

    @staticmethod
    def check_conflict(j1,j2s):
        flag = False

        for j2 in j2s:
            #j1 start earlier than j2
            if j2.d-j2.p < j1.r+j1.p and j1.d-j1.p < j2.r+j2.p:
                flag = True
                break
        
        return flag

class jobs():
    def __init__(self,num,term,max_p):
        self.j = []

        for i in range(num):
            conflict_flag = True
            while conflict_flag:
                tmp_j = job(term,max_p)
                conflict_flag = job.check_conflict(tmp_j,self.j)

            self.j.append(tmp_j)

    def print(self):
        for idx,j in enumerate(self.j):
            print(f"job {idx},",end="")
            j.print()

    def get_r(self,i):
        return self.j[i].r

    def get_p(self,i):
        return self.j[i].p

    def get_d(self,i):
        return self.j[i].d


        
if __name__ == "__main__":
    #Setting Random Seed
    seed(1)

    #Job number
    V = 20
    TERM = 100

    #Define Problem
    myjobs = jobs(V,TERM,5)
    myjobs.print()

    #Define Pulp Problem 
    p = LpProblem("OneMachineProblem")

    ##########################
    #Define Variables

    #Order between two Jobs
    #x[i][j] = 1 mean job i start earlier than job j 
    #x[i][j] = 0 mean job i start later than job j 
    x = LpVariable.dict("x",indexs=(range(V),range(V)),lowBound=0,upBound=1,cat=LpBinary)

    #End time of jobs
    c = LpVariable.dict("c",indexs=(range(V)),lowBound=0,upBound=TERM,cat=LpInteger)

    #to minimize end
    y = LpVariable("y",lowBound=0,upBound=TERM,cat=LpInteger)

    ##########################
    #Ojbective is minimize end time
    p += y

    ##########################
    #subject to...

    #define y is later than all end time(c)
    for i in range(V):
        p += c[i] <= y

    #Job End timing(c) is after ready time(r)+producing time(p)
    for i in range(V):
        p += c[i] >= myjobs.get_r(i) + myjobs.get_p(i)
    
    #Job End tiing(c) is earlier than delivery time(d)
    for i in range(V):
        p += c[i] <= myjobs.get_d(i)

    # #2 Jobs Condition    
    for i in range(V):
        for j in range(i+1,V):
            p += c[i] <= c[j] - myjobs.get_p(j) + TERM*(1-x[(i,j)])

    for i in range(V):
        for j in range(i+1,V):
            p += c[j] <= c[i] - myjobs.get_p(i) + TERM*x[(i,j)]
    
    print("\nSolving...")
    solver=PULP_CBC_CMD(msg=0)
    stat=p.solve(solver)
    print(f"  Done: {LpStatus[stat]}")


    if stat == 1:
        print("\nResult:")
        for i in range(V):
            print(f"job {i}, {c[i].value() - myjobs.get_p(i) } , {c[i].value()}")

        print(f"Best Delivery = {y.value()}")