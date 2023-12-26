import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

# 加载数据集
print("Loading 20 newsgroups dataset...")
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# 数据预处理
print("Preprocessing data...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target
X_test = vectorizer.transform(newsgroups_test.data)
y_test = newsgroups_test.target

# 划分训练集和验证集
print("Splitting training and validation set...")
split_idx = int(len(X_train.toarray()) * 0.8)
X_val = X_train.toarray()[split_idx:]
y_val = y_train[split_idx:]
X_train = X_train.toarray()[:split_idx]
y_train = y_train[:split_idx]

# 定义KNN分类器
class KNN():
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        distances = []
        for i in range(len(self.X_train)):
            distance = np.sqrt(np.sum(np.square(self.X_train[i] - X_test)))
            distances.append((distance, self.y_train[i]))

        distances = sorted(distances)[:self.k]
        labels = [label for _, label in distances]
        return max(set(labels), key=labels.count)

# 训练模型并进行可视化敏感分析
print("Training model and performing hyperparameter tuning...")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
f1_scores = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    y_pred = []
    for i in range(len(X_val)):
        pred = knn.predict(X_val[i])
        y_pred.append(pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
    f1_scores.append(f1)

# 可视化敏感分析结果
import matplotlib.pyplot as plt

plt.plot(k_values, f1_scores)
plt.xlabel('k')
plt.ylabel('F1 score')
plt.title('F1 score vs. k')
plt.show()

# 在测试集上评估模型表现
print("Evaluating model on test set...")
k_best = k_values[np.argmax(f1_scores)]
knn = KNN(k=k_best)
knn.fit(X_train, y_train)

y_pred = []
for i in range(len(X_test)):
    pred = knn.predict(X_test[i])
    y_pred.append(pred)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")
