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
    def __init__(self, city_num):
        self.PATH=[]
        self.CHOICE = [i for i in range(city_num)]
        self.LENGTH=0
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

        self.LENGTH+=dist[self.PATH[-1]][self.PATH[0]]
        self.PATH.append(init)
        return self.LENGTH,self.PATH

class TSP:
    def __init__(self, yaml_file, city_pos):
        f=open(yaml_file,'r',encoding='utf8')
        result=f.read()
        f.close()
        dict=yaml.load(result, Loader=yaml.FullLoader)
        self.ITER_NUM=0
        self.CITIES=city_pos
        self.CITY_NUM=len(city_pos)
        self.CITY_SERIES=[i for i in range(self.CITY_NUM)]
        self.ANT_NUM=dict['ANT_NUM']

        ##following is the hyperparameter
        self.rho=dict['rho']
        self.alpha=dict['alpha']
        self.beta=dict['beta']
        self.lr=dict['lr']
        self.lg=dict['lg']        

        ##distance matrix
        self.ADJ_MATRIX=np.zeros((self.CITY_NUM, self.CITY_NUM), dtype=np.float64)
        for i in range(self.CITY_NUM):
            for j in range(i+1, self.CITY_NUM):
                self.ADJ_MATRIX[i][j]=np.linalg.norm(city_pos[i]-city_pos[j])
        self.ADJ_MATRIX=self.ADJ_MATRIX+self.ADJ_MATRIX.T

        ##Pheromone matrix
        self.MU_MATRIX=(np.ones((self.CITY_NUM, self.CITY_NUM), dtype=np.float64)-np.diag(np.ones(self.CITY_NUM)))*\
                    self.CITY_NUM*(self.CITY_NUM-1)/(np.sum(self.ADJ_MATRIX))

        ##Inspiring matrix
        self.INS_MATRIX=np.where(self.ADJ_MATRIX==0,0,1/self.ADJ_MATRIX)

        ##Here is the result
        self.best_length=np.inf
        self.best_path=[]
        self.history_length=[]
        self.history_best=[]

    def set_lg(self,lg):
        self.lg=lg

    def random_start(self):
        if self.ANT_NUM <= self.CITY_NUM:  # 蚂蚁数 <= 城市数
            self.INITIAL_START = np.random.permutation(range(self.CITY_NUM))[:self.ANT_NUM]
        else:  # 蚂蚁数 > 城市数
            self.INITIAL_START = np.random.randint(self.CITY_NUM, size=self.ANT_NUM)
            self.INITIAL_START[:self.CITY_NUM] = np.random.permutation(range(self.CITY_NUM))

    def get_transfer(self):
        self.TRANS_MATRIX=(self.MU_MATRIX**self.alpha)*(self.INS_MATRIX**self.beta)

    def mu_update(self, f, t, iter_best_reciprocal=0.0):
        updated_value=(1-self.rho)*self.MU_MATRIX[f][t]+\
                        self.rho*(iter_best_reciprocal*100+self.lr*np.max(self.MU_MATRIX[t]))
        self.MU_MATRIX[f][t]=self.MU_MATRIX[t][f]=updated_value

    def iter(self):
        ##First init the delta phe and trans matrix at the begining of every iter
        ##Assign random start for each ant
        self.get_transfer()
        self.random_start()

        length_list=[] # record the total length each ant walks
        path_list=[]  # record the path each ant travels

        for i in range(self.ANT_NUM):
            temp_ant=Ant(self.CITY_NUM)
            length, path=temp_ant.walk_through(self.TRANS_MATRIX,self.ADJ_MATRIX,self.INITIAL_START[i])
            length_list.append(length)
            path_list.append(path)

            ##Local update
            for i in range(self.CITY_NUM):
                self.mu_update(path[i],path[i+1])
            self.get_transfer()
        index=int(np.argmin(length_list))
        self.history_length.append(length_list)
        if length_list[index] < self.best_length:
            self.history_best.append(length_list[index])
            self.best_length = length_list[index]
            self.best_path = path_list[index]
        else:
            self.history_best.append(self.best_length)

        ##Global update
        for i in range(self.CITY_NUM):
            ## Iter best based
            if self.lg==1:
                self.mu_update(path_list[index][i],path_list[index][i+1],1/length_list[index])
            ## global best based
            else:
                self.mu_update(self.best_path[i],self.best_path[i+1],1/self.best_length)
        return length_list

    def solve(self,num_iter):
        start=time.time()
        self.ITER_NUM=num_iter
        for i in range(num_iter):
            length_list = self.iter()
        print(time.time()-start)
        print(np.mean(self.MU_MATRIX))

    def visualize(self):
        plt.figure(figsize=(6, 10))
        plt.subplot(211)
        plt.plot(self.history_length, 'b.')
        plt.plot(self.history_best, 'r-', label='history_best')
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
        plt.show()
        plt.close()
        print(self.best_length)

###-----------------------------  Ant Func  -----------------------------###
