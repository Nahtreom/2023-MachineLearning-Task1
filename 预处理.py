from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import string

# 加载数据集并进行数据清洗
newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroups_train.data)

# 对每个文档进行数据清洗和统计
doc_word_counts = []
for i in range(X.shape[0]):
    # 获取当前文档的词项列表
    words = newsgroups_train.data[i].split()
    words_dict = dict.fromkeys(words, 0)
    # 对词项列表进行数据清洗

    words_dict_clean = {}
    for word in words_dict.keys():
        word = word.lower() # 大写字符转换为小写字符
        word = word.translate(str.maketrans("", "", string.punctuation)) # 去掉标点符号
        if word in vectorizer.vocabulary_.keys(): # 检查单词是否在词汇表中
            words_dict_clean[word] = newsgroups_train.data[i].count(word)
    # 将当前文档的词项及其出现次数存储到doc_word_counts列表中
    doc_word_counts.append(words_dict_clean)

# 构建稀疏矩阵
rows = []
cols = []
data = []
for i, doc in enumerate(doc_word_counts):
    for j, count in doc.items():
        rows.append(i)
        cols.append(vectorizer.vocabulary_[j])
        data.append(count)
sparse_matrix = csr_matrix((data, (rows, cols)), shape=X.shape)

# 可视化稀疏矩阵
plt.spy(sparse_matrix, aspect='auto',markersize=0.1)
plt.xlabel('Word Index')
plt.ylabel('Document Index')
plt.show()