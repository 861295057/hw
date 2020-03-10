import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class PerceptronDuality():
    def __init__(self):
        self.learning_rate = 0.00001
        self.max_iteration = 100
        self.alpha = None
        self.bias = None
        self.W = None
    
    def train(self, features, labels):    
        num_samples,_ = features.shape
        self.alpha = np.zeros(num_samples)
        self.bias = 0
        print("argument initialize.")
        gram = np.dot(features, features.T)
        #计算gram矩阵
        print("gram caculate finished")
        correct_count = 0
        time = 0
        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1  )
            #print(index)
            inner_product = gram[index]
            y =2 * labels[index] - 1
            wx = y * ( np.sum(self.alpha * labels * inner_product) + self.bias)
            #print(np.sum(self.alpha * labels * inner_product) + self.bias)
            # * means product
            if wx > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            self.alpha[index] = self.alpha[index] + self.learning_rate
            
            self.bias = self.bias + self.learning_rate * y
        self.W = np.sum(self.alpha * labels * features.T,axis = 1)
        print(self.W.shape,labels.shape,features.T.shape)
        #变为行向量
        # detials axis = ?    
    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            prd = sum(x * self.W) + self.bias
            labels.append(int(prd > 0))
        return labels




print('Start read data')

#data input
raw_data = pd.read_csv('../data/train_binary.csv', header=0)
data = raw_data.values #(420000,785)
imgs = data[0:6000:, 1::]  #（420000,784）
labels = data[0:6000:, 0]

# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
#test_features.shape = (13860,784)
#test_labels.shape = (13860,) means 行向量，否则列向量(13860,1)

print ('Start training')
p = PerceptronDuality()
p.train(train_features, train_labels)
print('Start predicting')
test_predict = p.predict(test_features)
score = accuracy_score(test_labels, test_predict)
print("The accruacy socre is ", score)

