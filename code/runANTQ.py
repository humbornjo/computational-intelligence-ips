import antq
import mmas
import antq_non as antn
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

pos=antq.tsp2numpy(antq.read_TSP('berlin52.tsp'))

tsp=antq.TSP('./antq_config.yaml',pos)

tsp.solve(200)

#tsp.visualize()
print(tsp.last_change,tsp.CURRENT_ITER,tsp.best_length,tsp.time_cost)

# res=[]
# t=10
# for ii in range(t): 
#     tsp=ant.TSP('./antq_config.yaml',pos)
#     tsp.solve(200)
    
#     res.append(tsp.best_length)

# label=np.linspace(1,t,t)
# plt.scatter(label,res)

# for x, y in zip(label,res):
#     plt.text(x,y+0.05,'%.2f'%y)

# plt.savefig('ANTQ_opted.png', dpi=500)
# res=[]
# for i in range(2):
#     temp=[]
#     for ii in range(10):
#         tsp=antq.TSP('./antq_config.yaml',pos)
#         if i==1:
#             tsp.set_lg(0)
#         else:
#             pass
#         tsp.solve(200)
#         print(tsp.time_cost)

#         temp.append(tsp.best_length)

#     res.append(temp)

# label=['iter','glob']
# plt.boxplot(res,labels=label)
# plt.savefig('comp_ig.png', dpi=500)

# ant_q=[]

# for ii in range(10):
#     tsp=antq.TSP('./antq_config.yaml',pos)
#     tsp.solve(200)

#     ant_q.append(tsp.time_cost)

# res.append(ant_q)

# mma_s=[]

# for ii in range(10):
#     tsp=mmas.TSP('./mmas_config.yaml',pos)
#     tsp.solve(200)

#     mma_s.append(tsp.time_cost)

# res.append(mma_s)

# label=['Ant-Q','MMAS']
# plt.boxplot(res,labels=label)
# plt.savefig('comp_time.png', dpi=500)


# res=[]

# temp=[]
# for ii in range(10):
#     tsp=antq.TSP('./antq_config.yaml',pos)
#     tsp.solve(200)
#     temp.append(tsp.best_length)

# res.append(temp)

# temp=[]
# for ii in range(10):
#     tsp=antn.TSP('./antq_config.yaml',pos)
#     tsp.solve(200)
#     temp.append(tsp.best_length)

# res.append(temp)

# label=['cast','non-cast']
# plt.boxplot(res,labels=label)
# plt.savefig('comp_cast.png', dpi=500)

