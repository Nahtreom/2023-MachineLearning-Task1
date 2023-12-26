import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 数据预处理
def load_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target

    return X_train, y_train, X_test, y_test

def preprocess(X_train, X_test):
    X_train = csr_matrix(X_train.astype(np.float32).toarray())
    X_test = csr_matrix(X_test.astype(np.float32).toarray())

    return X_train, X_test

# PyTorch数据加载
def create_dataset(X, y):
    X_tensor = torch.from_numpy(X.toarray()).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

def get_data_loaders(X_train, y_train, X_valid, y_valid, batch_size):
    train_dataset = create_dataset(X_train, y_train)
    valid_dataset = create_dataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2)

    return train_loader, valid_loader

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# 定义训练函数
def train(net, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        net.train()
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for inputs, targets in valid_loader:
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true += targets.tolist()
                y_pred += predicted.tolist()

            f1_macro = f1_score(y_true, y_pred, average='macro')
            print('Epoch [{}/{}], F1 macro: {:.4f}'.format(epoch+1, num_epochs, f1_macro))

# 定义主函数
if __name__ == '__main__':
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data()
    X_train_raw, X_test_raw = preprocess(X_train_raw, X_test_raw)

    # 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw, test_size=0.2,
                                                          stratify=y_train_raw, random_state=42)

    # 加载数据
    BATCH_SIZE = 32
    train_loader, valid_loader = get_data_loaders(X_train, y_train, X_valid, y_valid, BATCH_SIZE)

    # 构建模型
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train_raw))
    hidden_dim = 100
    lr = 0.01
    num_epochs = 10

    net = Net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    # 训练模型
    train(net, train_loader, valid_loader, criterion, optimizer, num_epochs)

    # 在测试集上评估模型性能
    X_test = csr_matrix(X_test_raw.astype(np.float32).toarray())
    y_test = y_test_raw
    test_dataset = create_dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
    net.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for inputs, targets in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true += targets.tolist()
            y_pred += predicted.tolist()

        f1_macro = f1_score(y_true, y_pred, average='macro')
        print('Test F1 macro: {:.4f}'.format(f1_macro))