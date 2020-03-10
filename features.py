import pandas as pd
import numpy as np
import random
from cv2 import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class PerceptronDuality():
    def __init__(self):
        self.learning_rate = 0.00001
        self.max_iteration = 50000
        self.alpha = None
        self.bias = None
        self.W = None
    
    def train(self, features, labels):    
        num_samples,_ = features.shape
        labels = 2 * np.array(labels) - 1
        self.alpha = np.zeros((num_samples,))
        self.bias = 0
        print("argument initialize.")
        gram = np.dot(features, features.T)
        print("gram caculate finished")
        correct_count = 0
        time = 0
        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            inner_product = gram[index]
            y = labels[index]
            wx = y * ( np.sum(self.alpha * labels * inner_product) + self.bias)
            # * means product
            if wx > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                
                continue
            
            self.alpha[index] = self.alpha[index] + self.learning_rate
            self.bias = self.bias + self.learning_rate * y
        self.W = np.sum(self.alpha * labels * features.T,axis = 1)
        # detials axis = ?    
    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            prd = sum(x * self.W) + self.bias
            labels.append(int(prd > 0))
            #labels.append(np.sign(np.dot(x, self.W) + self.bias))
        return labels
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features




print('Start read data')

#data input
raw_data = pd.read_csv('../data/train_binary.csv', header=0)
data = raw_data.values #(420000,785)
imgs = data[0:42000:, 1::]  #（420000,784）
labels = data[0:42000:, 0]
fx = get_hog_features(imgs)
# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(fx, labels, test_size=0.33, random_state=23323)
#test_features.shape = (13860,784)
#test_labels.shape = (13860,) means 行向量，否则列向量(13860,1)

print ('Start training')
p = PerceptronDuality()
p.train(train_features, train_labels)
print('Start predicting')
test_predict = p.predict(test_features)
score = accuracy_score(test_labels, test_predict)
print("The accruacy socre is ", score)

