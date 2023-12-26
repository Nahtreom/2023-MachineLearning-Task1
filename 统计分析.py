import os
import tarfile
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter

# 下载并加载数据集
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# 打印数据集的统计信息
print('训练集大小:', len(newsgroups_train.data))
print('测试集大小:', len(newsgroups_test.data))
print('类别数:', len(newsgroups_train.target_names))

# 统计每个类别中新闻帖子的数量
train_count = Counter(newsgroups_train.target)
test_count = Counter(newsgroups_test.target)
for i, target_name in enumerate(newsgroups_train.target_names):
    print('类别{}({})包含{}个训练样本和{}个测试样本'.format(i, target_name, train_count[i], test_count[i]))

# 解压bydate版本的数据集
tar_file = tarfile.open('20news-bydate.tar.gz', 'r:gz')
tar_file.extractall(path='D:\py\wwz')
tar_file.close()

# 打印bydate版本数据集的统计信息
path1 = "D:/py/wwz/20news-bydate-test"
folders1 = os.listdir(path1)
path2 = "D:/py/wwz/20news-bydate-train"
folders2 = os.listdir(path2)
print('bydate版本测试数据集包含{}个文件夹'.format(len(folders1)))
for folder1 in folders1:
    files = os.listdir(os.path.join(path1, folder1))
    print('文件夹{}包含{}个文本文件'.format(folder1, len(files)))
print('bydate版本训练数据集包含{}个文件夹'.format(len(folders2)))
for folder2 in folders2:
    files = os.listdir(os.path.join(path1, folder2))
    print('文件夹{}包含{}个文本文件'.format(folder2, len(files)))

# 随机选择一些新闻帖子并打印它们的文本内容
#sample_indices = np.random.choice(len(newsgroups_train.data), size=5, replace=False)
#for i in sample_indices:
#    print('类别:', newsgroups_train.target_names[newsgroups_train.target[i]])
#   print('文本内容:', newsgroups_train.data[i])
#   print('-------------------------------------------')