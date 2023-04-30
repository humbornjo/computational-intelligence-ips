import mmas as ant
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

pos=ant.tsp2numpy(ant.read_TSP('berlin52.tsp'))

# tsp=ant.TSP('./mmas_config.yaml',pos)

# tsp.solve(200)

# tsp.visualize()
# print(tsp.last_change,tsp.CURRENT_ITER,tsp.best_length,tsp.time_cost)


# res=[]
# for i in range(5):
#     temp=[]
#     for ii in range(10):
#         tsp=ant.TSP('./mmas_config.yaml',pos)
#         tsp.set_alpha(i+1)
#         tsp.solve(400)

#         temp.append(tsp.best_length)

#     res.append(temp)

# label=['1','2','3','4','5']
# plt.boxplot(res,labels=label)
# plt.savefig('MMAS_alpha.png', dpi=500)

res=[]
for i in range(2):
    temp=[]
    for ii in range(10):
        tsp=ant.TSP('./mmas_config.yaml',pos)
        if i==1:
            tsp.set_stable(1)
        else:
            tsp.set_stable(0)
        tsp.solve(200)
        print(tsp.best_length)

        temp.append(tsp.time_cost)

    res.append(temp)

label=['improved','non']
plt.boxplot(res,labels=label)
plt.savefig('comp_cons.png', dpi=500)


