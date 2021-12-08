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






#


class Detection:

    def __init__(self):
        self.data_path = 'F://MH//Data//Anomaly//detection//'

        self.car_list = [6, 8, 9, 11, 14, 16, 17, 19]

        self.date = None
        self.carno = None
        self.n_features = 9
        self.n_car = 8
        self.n_motorINcar = 4
        self.epsilon_of_channel = None
        self.epsilon_of_score = -1.1
        self.fillnan =True





    def Dataloader(self, dirpath):

        self.date = dir[-4:]
        filepath = dirpath + '//6.csv'
        if not os.path.exists(filepath):
            print("不存在"+filepath)
            return []



        car = pd.read_csv(filepath, encoding='utf-8', index_col=0)
        car.dropna(axis=0, how="all", inplace=True)
        if self.fillnan:
            car.fillna(method="ffill", inplace=True)

        # car.fillna(method="bfill", inplace=True)
        #  将索引截断 不同车数据中存在 08：10：00 和08：10：01  两表合并时不能对应
        car.index = car.index.map(lambda x: x[:-2])
        print(car.shape)


        for no in self.car_list[1:]:
            filepath = dirpath + '//{}.csv'.format(no)

            if not os.path.exists(filepath):
                continue
            data = pd.read_csv(filepath, encoding='utf-8', index_col=0)
            data.dropna(axis=0, how="all", inplace=True)

            if self.fillnan:
                data.fillna(method="ffill", inplace=True)
            # data.fillna(method="bfill")

            data.index = data.index.map(lambda x: x[:-2])
            car = pd.merge(car, data, left_index=True, right_index=True, how = "inner")

        car = car.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
        # 去除重复索引   1、将原行索引 保存为新列  2、根据新列去除重复值 3、将新列设置为行索引
        print(car.shape)
        return car

    def Channel_samples(self, data):
        channel_words = ["GM侧温度", "GW侧温度",
                         "PM侧温度", "PW侧温度",
                         "1位轴温", "2位轴温",
                         "驱动端", "非驱动端", "定子"]


        channels = []
        channel_mean = []
        col = data.columns

        for w in channel_words:
            idx = [i for i, x in enumerate(col) if w in x]
            if w == "驱动端":
                idx = [i for i, x in enumerate(col) if w in x and "非" not in x]
            channel = data.iloc[:, idx]

            channels.append(channel)
            channel_mean.append(channel.mean(axis=1).mean())

        samples = np.concatenate(channels, axis=1)

        if not self.epsilon_of_channel: #  各个通道的阈值根据首批数据确定，也可以人工设置
            self.epsilon_of_channel = channel_mean


        return samples

    def anomaly_score(self, samples):
        # scores:  a matrix with shape (32, time)   32(32台电机) = 8car*4
        # attr_scores : like scores  judge with indivial rules
        #

        scores = []
        attr_scores = []

        for t in range(len(samples)):
            sample = samples[t].reshape(-1, 32).T

            # define model
            model = LocalOutlierFactor(n_neighbors=16, contamination=0.1, metric="euclidean")
            y = model.fit_predict(sample).reshape(1, -1)
            score = model.negative_outlier_factor_.reshape(1, -1)


            # 人工规则
            avg_value = sample.mean(axis=0)
            # a matrix of 0 an 1 , 1 表示该位置处的数值满足人工剔除规则
            score_manual = np.zeros_like(sample)

            # 数值  1、大于时间维度阈值 2、大于同一时刻下所有相同类型数值的均值
            for index in range(sample.shape[1]):
                for j in range(sample.shape[0]):
                    score_manual[j, index] = 1 if sample[j, index] > avg_value[index] and \
                                                  sample[j, index] > 1.4 * self.epsilon_of_channel[index] else 0  # 筛选

            attr_score = score_manual[:, :].mean(axis=1)  # 统计每个样本中9个维度值有多少满足人工规则
            for i in range(len(attr_score)):
                attr_score[i] = 1 if attr_score[i] > 0.4 else 0


            scores.append(score)
            attr_scores.append(attr_score.reshape(1, 32))

        scores = np.concatenate(scores, axis=0)
        attr_scores = np.concatenate(attr_scores,axis =0 )
        print("----------date:{}--------".format(self.date))
        print("中位数：{}， 平均值：{}， 标准差{}，最大值：{}，最小值{}, 80%分位数：{}".format(
            np.median(scores),
            np.mean(scores),
            np.std(scores),
            np.amax(scores),
            np.amin(scores),
            np.percentile(scores,20)
        ))


        scores[scores > -1.1] = 0
        scores[scores < -1.1] = 1



        return scores, scores*attr_scores



    def plot_error(self, scores, scores_withRules):


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # for ax, interp_method in zip(axs.flat, methods):
        # ax.imshow(scores.T, interpolation=None, cmap='viridis',aspect='auto')
        ax.pcolor(scores.T, edgecolors='k', cmap='viridis', linewidths=0.2 , label ="LOF" )
        ax2.pcolor(scores_withRules.T, edgecolors='k', cmap='viridis', linewidths=0.2 , label ="LOF with Rules" )

        plt.title("score of {}".format(self.date))

        plt.legend()
        plt.tight_layout()
        plt.show()

    def nothing (self):
        model = LocalOutlierFactor(n_neighbors=16, contamination=0.01)
        # 返回基于规则判断的矩阵



        sample = samples[0].reshape(-1,32).T

        y = model.fit_predict(sample).reshape(1, -1)

        score = model.negative_outlier_factor_.reshape(1, -1)

        avg_value = sample.mean(axis=0)

        score_manual = np.zeros_like(sample)








if __name__ == "__main__":

    test = Detection()

    data_path = 'F://MH//Data//Anomaly//detection//'
    for root, dirs, files in os.walk(data_path):

        for dir in dirs:
            dirpath = os.path.join(root, dir)

            car = test.Dataloader(dirpath)
            if len(car) == 0:
                continue

            samples = test.Channel_samples(car)


            raw_scores, score_withRules = test.anomaly_score(samples)

            # test.plot_error(raw_scores, score_withRules)





            # for i in range(9):
            #
            #     plt.figure(figsize=(20, 4))
            #
            #     value = samples[:, i*32:(i+1)*32]
            #     plt.plot(value, color='green', alpha=0.5)
            #     for j in range(0, 32):
            #         x = np.where(scores[:, j] == 1)
            #         plt.scatter(x, scores[x, j] * value[x, j], c='red', marker='*')
            #
            #     plt.show()
            #
            # for i in range(0, 32):
            #     x = np.where(scores[:, i])
            #     plt.scatter(x, scores[x, i], marker='*')
            # plt.show()





'''
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

    alpha = [60, 20, 40, 20] #  -stator , nondrive , drive  ,diff 四个维度的下限

    for index in range(sample.shape[1]):
        for j in range(sample.shape[0]):
            sample[j, index] = 1 if sample[j, index] > avg_value[index] and \
                                    sample[j, index] > alpha[index] else 0 # 筛选出大于测量值均值的点

    attr_score = sample[:,:3].mean(axis=1)  # 统计每个样本中三个维度值有多少满足人工规则
    for i in range(len(attr_score)):
        attr_score[i] = 1 if attr_score[i] > 0 else 0

    # score = score * attr_score
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

'''


















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