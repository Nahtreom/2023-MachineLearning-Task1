import string
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# 加载20Newsgroups数据集
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# 将原始词条去掉标点符号，大写字符转换为小写字符
translator = str.maketrans('', '', string.punctuation)
docs = [doc.lower().translate(translator) for doc in newsgroups.data]

# 划分训练集和测试集
train_docs, test_docs, train_labels, test_labels = train_test_split(docs, newsgroups.target, test_size=0.2, random_state=42)

# 将训练集划分成训练集和验证集
train_docs, val_docs, train_labels, val_labels = train_test_split(train_docs, train_labels, test_size=0.2, random_state=42)

# 统计每个词项在属于该类文档的总次数
class_word_counts = []
for i in range(len(newsgroups.target_names)):
    class_docs = [train_docs[j] for j in range(len(train_docs)) if train_labels[j] == i]
    word_counts = {}
    for doc in class_docs:
        for word in doc.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    class_word_counts.append(word_counts)

# 统计每个类的先验概率
class_prior_probs = np.array([np.sum(train_labels == i) / len(train_labels) for i in range(len(newsgroups.target_names))])

# 预测测试集的标签
predicted_labels = []
for doc in test_docs:
    log_probs = np.zeros(len(newsgroups.target_names))
    for i in range(len(newsgroups.target_names)):
        log_prob = np.log(class_prior_probs[i])
        word_counts = class_word_counts[i]
        for word in doc.split():
            if word in word_counts:
                log_prob += np.log((word_counts[word] + 1) / (sum(word_counts.values()) + len(word_counts)))
        log_probs[i] = log_prob
    predicted_labels.append(np.argmax(log_probs))

# 评估模型
precision = precision_score(test_labels, predicted_labels, average='macro')
recall = recall_score(test_labels, predicted_labels, average='macro')
f1 = f1_score(test_labels, predicted_labels, average='macro')
print('Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))

# 可视化超参数对F1的影响
alphas = np.arange(0.01, 2.01, 0.1)
f1s = []
for alpha in alphas:
    class_word_counts = []
    for i in range(len(newsgroups.target_names)):
        class_docs = [train_docs[j] for j in range(len(train_docs)) if train_labels[j] == i]
        word_counts = {}
        for doc in class_docs:
            for word in doc.split():
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
        class_word_counts.append(word_counts)

    class_prior_probs = np.array([np.sum(train_labels == i) / len(train_labels) for i in range(len(newsgroups.target_names))])

    predicted_labels = []
    for doc in val_docs:
        log_probs = np.zeros(len(newsgroups.target_names))
        for i in range(len(newsgroups.target_names)):
            log_prob = np.log(class_prior_probs[i])
            word_counts = class_word_counts[i]
            for word in doc.split():
                if word in word_counts:
                    log_prob += np.log((word_counts[word] + alpha) / (sum(word_counts.values()) + alpha * len(word_counts)))
                else:
                    log_prob += np.log(alpha / (sum(word_counts.values()) + alpha * len(word_counts)))
            log_probs[i] = log_prob
        predicted_labels.append(np.argmax(log_probs))

    f1s.append(f1_score(val_labels, predicted_labels, average='macro'))

plt.plot(alphas,f1s)
plt.xlabel('Alpha')
plt.ylabel('F1 score')
plt.title('Effect of smoothing parameter on F1 score')
plt.show()