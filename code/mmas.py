import tsplib95, yaml, time
import numpy as np
import matplotlib.pyplot as plt

###----------------------------- Basic Func -----------------------------###
def read_TSP(path):
    problem = tsplib95.load(path).as_name_dict()
    return problem

def tsp2numpy(tsp_dict):
    pos_list=[]
    for value in tsp_dict['node_coords'].values():
        pos_list.append(value)
    return np.array(pos_list,dtype=np.float64)

###----------------------------- Basic Func -----------------------------###



###-----------------------------  Ant Func  -----------------------------###
class Ant:
    def __init__(self, city_num, Q):
        self.PATH=[]
        self.CHOICE = [i for i in range(city_num)]
        self.LENGTH=0
        self.Q=Q
        self.DELTA=np.zeros((city_num, city_num))

    def walk_through(self,trans,dist,init):
        self.CHOICE.remove(init)
        self.PATH.append(init)
        while self.CHOICE:
            p=trans[self.PATH[-1]][self.CHOICE]
            p=p/np.sum(p)
            v=np.random.choice(self.CHOICE,p=p)
            self.CHOICE.remove(v)
            self.PATH.append(v)
            self.LENGTH+=dist[self.PATH[-2]][self.PATH[-1]]
            self.DELTA[self.PATH[-2]][self.PATH[-1]]=self.DELTA[self.PATH[-1]][self.PATH[-2]]=self.Q

        self.LENGTH+=dist[self.PATH[-1]][self.PATH[0]]
        self.DELTA[self.PATH[-1]][self.PATH[0]] = self.DELTA[self.PATH[0]][self.PATH[-1]] = self.Q
        self.PATH.append(init)
        self.DELTA/=self.LENGTH
        return self.LENGTH,self.PATH,self.DELTA

class TSP:
    def __init__(self, yaml_file, city_pos):
        f=open(yaml_file,'r',encoding='utf8')
        result=f.read()
        f.close()
        dict=yaml.load(result, Loader=yaml.FullLoader)
        self.ITER_NUM=0
        self.CURRENT_ITER=0
        self.CITIES=city_pos
        self.CITY_NUM=len(city_pos)
        self.CITY_SERIES=[i for i in range(self.CITY_NUM)]
        self.ANT_NUM=dict['ANT_NUM']

        ##following is the hyperparameter
        self.Q=dict['Q']
        self.rho=dict['rho']
        self.alpha=dict['alpha']
        self.beta=dict['beta']
        self.phe_up=dict['phe_up']
        self.phe_low=dict['phe_low']
        self.dumb=1
        self.stable=0

        ##distance matrix
        self.ADJ_MATRIX=np.zeros((self.CITY_NUM, self.CITY_NUM), dtype=np.float64)
        for i in range(self.CITY_NUM):
            for j in range(i+1, self.CITY_NUM):
                self.ADJ_MATRIX[i][j]=np.linalg.norm(city_pos[i]-city_pos[j])
        self.ADJ_MATRIX=self.ADJ_MATRIX+self.ADJ_MATRIX.T

        ##Pheromone matrix
        self.PHE_MATRIX=(np.ones((self.CITY_NUM, self.CITY_NUM), dtype=np.float64)-np.diag(np.ones(self.CITY_NUM)))*self.phe_up

        ##Inspiring matrix
        self.INS_MATRIX=np.where(self.ADJ_MATRIX==0,0,1/self.ADJ_MATRIX)

        ##Here is the result
        self.best_length=np.inf
        self.length_list=[]
        self.best_path=[]
        self.history_length=[]
        self.history_best=[]

    def set_alpha(self, alpha):
        self.alpha=alpha

    def set_beta(self, beta):
        self.beta=beta

    def set_rho(self, rho):
        self.rho=rho    

    def set_stable(self,stable):
        self.stable=stable
        
    def random_start(self):
        if self.ANT_NUM <= self.CITY_NUM:  # 蚂蚁数 <= 城市数
            self.INITIAL_START = np.random.permutation(range(self.CITY_NUM))[:self.ANT_NUM]
        else:  # 蚂蚁数 > 城市数
            self.INITIAL_START = np.random.randint(self.CITY_NUM, size=self.ANT_NUM)
            self.INITIAL_START[:self.CITY_NUM] = np.random.permutation(range(self.CITY_NUM))

    def get_transfer(self):
        self.TRANS_MATRIX=(self.PHE_MATRIX**self.alpha)*(self.INS_MATRIX**self.beta)

    def phe_update(self,delta):
        self.PHE_MATRIX*=(1-self.rho)
        self.PHE_MATRIX+=delta
        self.PHE_MATRIX=np.where(self.PHE_MATRIX>self.phe_up,self.phe_up,self.PHE_MATRIX)
        self.PHE_MATRIX=np.where(self.PHE_MATRIX<self.phe_low,self.phe_low,self.PHE_MATRIX)

    def opt_2(self, tour,length):
        flag=True
        while flag:
            flag=False
            for i in range(1,self.CITY_NUM):
                for j in range(i+1,self.CITY_NUM):
                    new_tour=tour[:i]+tour[i:j+1][::-1]+tour[j+1:]
                    new_length=0
                    for k in range(self.CITY_NUM):
                        new_length+=self.ADJ_MATRIX[new_tour[k]][new_tour[k+1]]
                    if new_length<length:
                        tour=new_tour
                        length=new_length
                        flag=True
                        break
                if flag:
                    break
        return tour, length

    def decline_ratio(self):
        return np.sqrt(1/self.dumb)*(self.ANT_NUM//20)

    def iter(self):
        ##First init the delta phe and trans matrix at the begining of every iter
        ##Assign random start for each ant
        self.DELTA_PHE = np.zeros((self.CITY_NUM, self.CITY_NUM), dtype=np.float64)
        self.get_transfer()
        self.random_start()

        length_list=[] # record the total length each ant walks
        path_list=[]  # record the path each ant travels

        for i in range(self.ANT_NUM):
            temp_ant=Ant(self.CITY_NUM,self.Q)
            length, path, delta=temp_ant.walk_through(self.TRANS_MATRIX,self.ADJ_MATRIX,self.INITIAL_START[i])
            length_list.append(length)
            path_list.append(path)
            self.DELTA_PHE+=delta

        index=int(np.argmin(length_list))

        self.length_list.append(length_list)

        opt_path,opt_length=self.opt_2(path_list[index],length_list[index])

        self.history_length.append(length_list)
        if opt_length < self.best_length:
            self.history_best.append(opt_length)
            self.best_length = opt_length
            self.best_path = opt_path
            self.last_change=self.CURRENT_ITER
            self.dumb=1
        else:
            self.history_best.append(self.best_length)
            bouns_delta=np.zeros((self.CITY_NUM,self.CITY_NUM))
            for i in range(self.CITY_NUM):
                if self.stable==1:
                    constant=self.ANT_NUM//20
                else:
                    constant=self.decline_ratio()
                bouns_delta[self.best_path[i]][self.best_path[i+1]]=bouns_delta[self.best_path[i+1]][self.best_path[i]]\
                    =self.Q*constant
            bouns_delta/=self.best_length
            self.DELTA_PHE+=bouns_delta
        self.phe_update(self.DELTA_PHE)
        return length_list

    def solve(self,num_iter):
        start=time.time()
        self.ITER_NUM=num_iter
        for i in range(num_iter):
            self.CURRENT_ITER+=1
            self.iter()
            if self.last_change/self.CURRENT_ITER<0.5:
                self.dumb+=1

                if self.dumb/self.ITER_NUM>0.1:
                    break
        self.time_cost=time.time()-start

    def visualize(self):
        plt.figure(figsize=(6, 10))
        plt.subplot(211)

        plt.plot(np.mean(self.length_list,axis=1), 'g-.',label='average length')
        plt.plot(np.min(self.length_list,axis=1), 'b-.',label='iteration best')
        plt.plot(self.history_best, 'r-', label='history best')
        plt.xlabel('Iteration')
        plt.ylabel('length')
        plt.legend()

        best_city=self.CITIES[self.best_path]
        plt.subplot(212)
        plt.plot(list(city[0] for city in best_city), list(city[1] for city in best_city), 'b-')
        plt.plot(list(city[0] for city in best_city), list(city[1] for city in best_city), 'r.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('ACO_'+str(np.around(self.best_length,4))+'.png', dpi=500)
        #plt.show()
        plt.close()

###-----------------------------  Ant Func  -----------------------------###
