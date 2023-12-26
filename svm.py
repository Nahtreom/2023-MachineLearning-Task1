from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import string
import matplotlib.pyplot as plt
from sklearn import metrics

# 加载数据集并划分数据集
newsgroups = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 数据预处理：去标点、转小写
translator = str.maketrans('', '', string.punctuation)  # 去除标点符号的翻译器
for i in range(len(X_train)):
    X_train[i] = X_train[i].translate(translator).lower()
for i in range(len(X_test)):
    X_test[i] = X_test[i].translate(translator).lower()
for i in range(len(X_val)):
    X_val[i] = X_val[i].translate(translator).lower()

# 特征提取：tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(X_val)

# 训练模型
clf = SVC(kernel='linear', C=1)
clf.fit(X_train_tfidf, y_train)

# 在验证集上评估模型
y_pred_val = clf.predict(X_val_tfidf)
print(classification_report(y_val, y_pred_val))

# 可视化超参数的敏感分析
Cs = [0.1, 1, 10, 100]
results = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train_tfidf, y_train)

    y_pred_val = clf.predict(X_val_tfidf)
    f1_macro = metrics.f1_score(y_val, y_pred_val, average='macro')
    results.append(f1_macro)

plt.plot(Cs, results)
plt.xlabel('C')
plt.ylabel('F1 macro')
plt.show()

# 在测试集上评估模型，给出precision和recall
y_pred_test = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_test))
