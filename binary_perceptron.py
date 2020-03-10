import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000
        self.w = None
        
    def train(self,features,labels):
        #self.w = np.random.randn((len(features[0]) + 1))
        self.w = [0.0] * (len(features[0]) + 1)
        #list 数乘扩展，+1含 b
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index =random.randint(0, len(labels)- 1 )
            x = list(features[index])
            #形成 w_hat
            x.append(1.0) # b variable
            y = 2 * labels[index] - 1 # +1 and -1
            #逐个元素处理
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
            
            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
                #update the arguments
            for i in range(len(self.w)):
                #include b
                self.w[i] += self.learning_step * (y * x[i])

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            #列表解析扩展
            labels.append(int(sum([self.w[j] * x[j] for j in range(len(self.w))]) > 0))
        return labels
print('Start read data')

#data input
raw_data = pd.read_csv('../data/train_binary.csv', header=0)
data = raw_data.values #(42000,785)
imgs = data[0::, 1::]  #（42000,784）
labels = data[::, 0]
# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
#test_features.shape = (13860,784)
#test_labels.shape = (13860,) means 行向量，否则列向量(13860,1)
print ('Start training')
p = Perceptron()
p.train(train_features, train_labels)
print('Start predicting')
test_predict = p.predict(test_features)
score = accuracy_score(test_labels, test_predict)
print("The accruacy socre is ", score)
