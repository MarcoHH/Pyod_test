import math


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import concat
from pandas import DataFrame
import os

from datetime import datetime as dt
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
import time
from sklearn.neighbors import kde
import numpy as np





data_path = 'F://MH//Data//experiment//data//detect//'
all_data =[]
car_list = []

for root, dirs, files in os.walk(data_path):
    stream =[]
    print(files)
    for file in files:

        filepath = os.path.join(root, file)
        data = np.load(filepath)[:,:12]



        # data = data.astype(np.float32)
        stream.append(data)



        date = filepath.split("\\", 1)[0][-4:]
        carno = filepath.split("\\", 1)[1][:-4]
    if stream:
        all_data.append(np.concatenate(stream, axis=1))



for data in all_data[:2]:
    car = pd.DataFrame(data).dropna(axis=0,how='any').values
    # for car in all_data:
    stator = car[:, [i*3 for i in range(32)]]
    non_drive = car[:, [i*3+1 for i in range(32)]]
    drive = car[:, [i*3+2 for i in range(32)]]
    diff = drive - non_drive
    model = LocalOutlierFactor(n_neighbors=16, contamination=0.01)

    scores = []
    for t in range(len(car)):

        sample = np.concatenate((stator[[t],:],non_drive[[t],:],drive[[t],:],diff[[t],:]),axis=1).reshape(-1,32).T

        y = model.fit_predict(sample).reshape(1,-1)


        score = model.negative_outlier_factor_.reshape(1,-1)
        avg_value = sample.mean(axis=0)

        alpha = [60, 20, 40, 20] # stator , nondrive , drive  ,diff 四个维度的下限

        for index in range(sample.shape[1]):
            for j in range(sample.shape[0]):
                sample[j, index] = 1 if sample[j, index] > avg_value[index] and \
                                        sample[j, index] > alpha[index] else 0 # 筛选出大于测量值均值的点

        attr_score = sample[:,:3].mean(axis=1)  # 统计每个样本中三个维度值有多少满足人工规则
        for i in range(len(attr_score)):
            attr_score[i] = 1 if attr_score[i] > 0 else 0

        score = score * attr_score
        score = score * 1


        scores.append(score)
          #定义一个LOF模型，异常比例是10%


    scores=np.concatenate(scores, axis=0)
    # result[result==1]=0
    scores[scores >= -1.6] = 0
    scores[scores < -1.6] = 1
    # scores = -scores







    # plt.figure(figsize=(20,4))
    #
    # plt.ylim(ymax=3,ymin=-3)
    # # x = [i for i in range(result.shape[0])]
    # for i in range(16,20):
    #     x = np.where(scores[:,i]==1)
    #     plt.scatter(x, scores[x, i])
    # plt.show()

    # avg = np.mean(car,axis=1)
    plt.figure(figsize=(20,8))
    plt.subplot(4,1,1)
    plt.plot(stator,color ='green',alpha=0.5)

    for i in range(0,32):
        x = np.where(scores[:, i] == 1)
        plt.scatter(x, scores[x, i] * stator[x, i], c='red', marker='*')

    plt.subplot(4,1,2)
    plt.plot(non_drive,color ='green',alpha=0.5)
    for i in range(0,32):
        x = np.where(scores[:, i])
        plt.scatter(x, scores[x, i] * non_drive[x, i], c='red', marker='*')
    plt.subplot(4,1,3)
    plt.plot(drive,color ='green',alpha=0.5)
    for i in range(0,32):
        x = np.where(scores[:, i])
        plt.scatter(x, scores[x, i] * drive[x, i], c='red', marker='*')
    plt.subplot(4,1,4)
    # plt.plot(diff,color ='green',alpha=0.5)
    # for i in range(0,32):
    #     x = np.where(scores[:, i])
    #     plt.scatter(x, scores[x, i], c='red', marker='*')
    # plt.plot(avg, linewidth=2,color ='black')
    plt.xlim(xmax=scores.shape[0], xmin=0)
    for i in range(0, 32):
        x = np.where(scores[:, i])
        plt.scatter(x,scores[x, i], marker='*')
    plt.show()
    # plt.savefig('')

# for car in all_data:
#
#     kpi_stator = []
#     plt.plot(car[:,:])
#     plt.show()
# plt.figure(figsize=(20,4))
# plt.plot(np.concatenate(all_data, axis=0))
# plt.show()

# from pylab import rcParams
# result = seasonal_decompose(stator, model='multiplicative', freq=4)
#
# rcParams['figure.figsize'] = 10, 5
# result.plot()

df = pd.DataFrame(stator[:,19])
cur = df.ewm(span=10).mean()

plt.plot(df)
plt.plot(cur)
plt.plot(df-cur)
plt.plot(df.values[1:]-df.values[0:-1])
plt.show()




















# 搜索序列中的拐点
'''
test = df.values[:,0]
count=0
res =[]
bias =0.2
sum_T = []
TTPII = []
T=10
for i in range(2, len(test)):
    test[i]=0.8*test[i]+0.2*test[i-1]

    if len(sum_T) == T:
        sum_T.pop(0)
        sum_T.append(test[i])
    else:
        sum_T.append(test[i])

    MT = (test[i] - sum(sum_T)/T) /(sum(sum_T)/T)
    if test[i] <= test[i - 1] > test[i - 2]+bias:
        count += 1
        res.append(i - 1)
        TTPII.append(MT)
        continue
    elif test[i]+bias < test[i - 1] >= test[i - 2]:
        count += 1
        res.append(i - 1)
        TTPII.append(MT)
        continue

    elif test[i] >= test[i - 1] < test[i - 2]-bias:
        count += 1
        res.append(i - 1)
        TTPII.append(MT)
        continue
    elif test[i]-bias >= test[i - 1] < test[i - 2]:
        count += 1
        res.append(i - 1)
        TTPII.append(MT)


plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(test)
plt.scatter(res, test[res], c='red', marker='*')
plt.subplot(2,1,2)
plt.plot(res,TTPII)
plt.show()


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
# for ax, interp_method in zip(axs.flat, methods):
# ax.imshow(scores.T, interpolation=None, cmap='viridis',aspect='auto')
c = ax.pcolor(scores.T, edgecolors='k', cmap='viridis', linewidths=0.2)


plt.tight_layout()
plt.show()

'''


#  核密度估计

'''
test = scores[:,18].reshape(-1,1)
X_plot = np.linspace(0, 10, 1000)[:, np.newaxis]

KDE = kde.KernelDensity(kernel='gaussian', bandwidth=0.4).fit(test)
a=KDE.score_samples(X_plot)
plt.plot(X_plot[:,0],np.exp(a),color='red', lw=2,
            linestyle='-')
plt.show()




plt.plot(test)
plt.show()
N=100
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
                    
                    
'''