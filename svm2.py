# 导入相关库
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 加载数据集，并进行数据预处理
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target
punctuations = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'''
for i in range(len(X)):
    X[i] = X[i].lower()
    for char in punctuations:
        X[i] = X[i].replace(char, ' ')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征表示：使用TF-IDF算法将文本数据转换为稀疏矩阵表示
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 划分验证集
X_train_tfidf, X_val_tfidf, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# 定义超参数范围
parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# 使用Grid Search进行模型超参数调优
clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5)
clf.fit(X_train_tfidf, y_train)

# 输出最优参数
print("Best parameters set found on training set:")
print(clf.best_params_)

# 在测试集上进行模型评估
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# 可视化展示模型超参数调优过程
C, gamma = np.meshgrid(parameters['C'], parameters['gamma'])
Z = clf.cv_results_['mean_test_score']
Z = Z.reshape(C.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(C, gamma, Z)
ax.set_xlabel('C')
ax.set_ylabel('gamma')
ax.set_zlabel('Mean Test Score')
plt.show()

# 计算并可视化展示Macro-averaging F1指标、Precision、Recall
target_names = newsgroups.target_names
print("Macro-averaging F1 score: {:.2f}".format(
    classification_report(y_test, y_pred, target_names=target_names, output_dict=True)['macro avg']['f1-score']))
print("Precision:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3)['precision'])
print("Recall:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3)['recall'])

# 可视化展示模型超参数对Macro-averaging F1指标的影响
f1_scores = []
for c in parameters['C']:
    scores = []
    for g in parameters['gamma']:
        clf = SVC(kernel='rbf', C=c, gamma=g)
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_val_tfidf)
        scores.append(classification_report(y_val, y_pred, target_names=target_names, output_dict=True)['macro avg']['f1-score'])
    f1_scores.append(scores)

    fig, ax = plt.subplots()
    cax = ax.imshow(f1_scores, interpolation='nearest', cmap=plt.cm.hot)
    ax.set_xticks(np.arange(len(parameters['gamma'])))
    ax.set_yticks(np.arange(len(parameters['C'])))
    ax.set_xticklabels(parameters['gamma'])
    ax.set_yticklabels(parameters['C'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(parameters['C'])):
        for j in range(len(parameters['gamma'])):
            ax.text(j, i, "{:.2f}".format(f1_scores[i][j]), ha="center", va="center", color="w")
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_title("Hyperparameter Tuning")

    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Macro-averaging F1-score', rotation=270, labelpad=15)

    plt.show()