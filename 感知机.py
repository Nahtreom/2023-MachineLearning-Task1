from sklearn.datasets import fetch_20newsgroups
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV

# 加载数据集
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# 定义标点符号列表
punctuation = set(string.punctuation)

# 数据预处理函数
def preprocess_data(data):
    preprocessed_data = []
    for document in data:
        # 去除标点符号并转换为小写字母
        words = ''.join([c for c in document.lower() if c not in punctuation])
        preprocessed_data.append(words)
    return preprocessed_data

# 对训练集和测试集进行数据预处理
X_train_raw = preprocess_data(newsgroups_train.data)
X_test_raw = preprocess_data(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# 分割出验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train, test_size=0.2, random_state=42)

# 定义TfidfVectorizer并提取训练集特征
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# 定义超参数搜索空间
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1]
eta_values = [0.0001, 0.001, 0.01, 0.1, 1]

# 定义Perceptron模型和超参数搜索空间
perceptron = Perceptron()
param_grid = {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': alpha_values, 'eta0': eta_values}

# 使用GridSearchCV选择最佳的超参数设置
grid_search = GridSearchCV(perceptron, param_grid, scoring='f1_macro', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳超参数设置
print('Best penalty:', grid_search.best_estimator_.get_params()['penalty'])
print('Best alpha:', grid_search.best_estimator_.get_params()['alpha'])
print('Best eta0:', grid_search.best_estimator_.get_params()['eta0'])

# 在最佳超参数设置下重新拟合整个训练集上的Perceptron模型
best_perceptron = Perceptron(penalty=grid_search.best_params_['penalty'], alpha=grid_search.best_params_['alpha'], eta0=grid_search.best_params_['eta0'])
best_perceptron.fit(X_train, y_train)

# 在验证集上进行预测并评估模型性能
X_valid = vectorizer.transform(X_valid)
y_pred_valid = best_perceptron.predict(X_valid)
f1_macro_valid = f1_score(y_valid, y_pred_valid, average='macro')
precision_macro_valid = precision_score(y_valid, y_pred_valid, average='macro')
recall_macro_valid = recall_score(y_valid, y_pred_valid, average='macro')
print('F1 Score on validation set (macro-averaging):', f1_macro_valid)
print('Precision on validation set (macro-averaging):', precision_macro_valid)
print('Recall on validation set (macro-averaging):', recall_macro_valid)

# 在测试集上进行预测并评估模型性能
X_test = vectorizer.transform(X_test_raw)
y_pred_test = best_perceptron.predict(X_test)
f1_macro_test = f1_score(y_test, y_pred_test, average='macro')
precision_macro_test = precision_score(y_test, y_pred_test, average='macro')
recall_macro_test = recall_score(y_test, y_pred_test, average='macro')
print('F1 Score on test set (macro-averaging):', f1_macro_test)
print('Precision on test set (macro-averaging):', precision_macro_test)
print('Recall on test set (macro-averaging):', recall_macro_test)

#绘制F1分数、精确度和召回率随着超参数变化的曲线
results = grid_search.cv_results_
alphas = np.array([params['alpha'] for params in results['params']])
etas = np.array([params['eta0'] for params in results['params']])
f1_scores = np.array(results['mean_test_score'])
precision_scores = np.array(results['mean_test_score'])
recall_scores = np.array(results['mean_test_score'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
ax1.scatter(alphas, f1_scores)
ax1.set_xlabel('Alpha')
ax1.set_ylabel('F1 Score')
ax2.scatter(etas, f1_scores)
ax2.set_xlabel('Eta')
ax2.set_ylabel('F1 Score')
ax3.scatter(alphas*etas, f1_scores)
ax3.set_xlabel('Alpha * Eta')
ax3.set_ylabel('F1 Score')
plt.show()