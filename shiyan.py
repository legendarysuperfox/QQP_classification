import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# --------------------------
# 1. 数据加载与预处理
# --------------------------
class QQPDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tfidf_matrix1, tfidf_matrix2):
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tfidf_matrix1 = tfidf_matrix1  # 问题1的TF-IDF特征
        self.tfidf_matrix2 = tfidf_matrix2  # 问题2的TF-IDF特征

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 转换为张量
        q1_tfidf = torch.tensor(self.tfidf_matrix1[idx].toarray()[0], dtype=torch.float32)
        q2_tfidf = torch.tensor(self.tfidf_matrix2[idx].toarray()[0], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return q1_tfidf, q2_tfidf, label


def load_and_preprocess_data(data_dir):
    # 加载TSV文件
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', on_bad_lines='skip')
        dev_df = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', on_bad_lines='skip')
        test_df = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', on_bad_lines='skip')
    except Exception as e:
        print(f"加载TSV文件失败: {str(e)}")
        return None  # 发生错误时返回None

    # 打印各数据集列名
    print("训练集列名：", train_df.columns.tolist())
    print("验证集列名：", dev_df.columns.tolist())
    print("测试集列名：", test_df.columns.tolist())

    # 提取标签（处理测试集无标签的情况）
    train_labels = train_df["is_duplicate"].fillna(0).tolist()
    dev_labels = dev_df["is_duplicate"].fillna(0).tolist()

    if "is_duplicate" in test_df.columns:
        test_labels = test_df["is_duplicate"].fillna(0).tolist()
    else:
        print("测试集无'is_duplicate'列，用0填充标签")
        test_labels = [0] * len(test_df)

    # 提取问题文本
    train_q1 = train_df["question1"].fillna("").tolist()
    train_q2 = train_df["question2"].fillna("").tolist()
    dev_q1 = dev_df["question1"].fillna("").tolist()
    dev_q2 = dev_df["question2"].fillna("").tolist()
    test_q1 = test_df["question1"].fillna("").tolist()
    test_q2 = test_df["question2"].fillna("").tolist()

    # 文本向量化：TF-IDF
    all_texts = train_q1 + train_q2 + dev_q1 + dev_q2 + test_q1 + test_q2
    count_vec = CountVectorizer(max_features=5000)
    count_vec.fit(all_texts)

    # 词频矩阵
    train_counts1 = count_vec.transform(train_q1)
    train_counts2 = count_vec.transform(train_q2)
    dev_counts1 = count_vec.transform(dev_q1)
    dev_counts2 = count_vec.transform(dev_q2)
    test_counts1 = count_vec.transform(test_q1)
    test_counts2 = count_vec.transform(test_q2)

    # TF-IDF转换
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(train_counts1 + train_counts2)  # 用训练集拟合

    train_tfidf1 = tfidf_transformer.transform(train_counts1)
    train_tfidf2 = tfidf_transformer.transform(train_counts2)
    dev_tfidf1 = tfidf_transformer.transform(dev_counts1)
    dev_tfidf2 = tfidf_transformer.transform(dev_counts2)
    test_tfidf1 = tfidf_transformer.transform(test_counts1)
    test_tfidf2 = tfidf_transformer.transform(test_counts2)

    # 创建数据集
    train_dataset = QQPDataset(train_q1, train_q2, train_labels, train_tfidf1, train_tfidf2)
    dev_dataset = QQPDataset(dev_q1, dev_q2, dev_labels, dev_tfidf1, dev_tfidf2)
    test_dataset = QQPDataset(test_q1, test_q2, test_labels, test_tfidf1, test_tfidf2)

    # 关键：确保返回5个变量（顺序一致）
    return train_dataset, dev_dataset, test_dataset, count_vec, tfidf_transformer

# --------------------------
# 2. 定义自定义模型
# --------------------------
class QQPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(QQPClassifier, self).__init__()
        # 输入维度：问题1的TF-IDF特征 + 问题2的TF-IDF特征 + 特征差的绝对值
        self.fc1 = nn.Linear(input_dim * 3, 256)  # 第一层全连接
        self.fc2 = nn.Linear(256, 128)  # 第二层
        self.fc3 = nn.Linear(128, 1)  # 输出层（二分类）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 防止过拟合

    def forward(self, q1, q2):
        # 融合两个问题的特征：拼接+差值
        diff = torch.abs(q1 - q2)  # 特征差值（捕捉差异）
        x = torch.cat([q1, q2, diff], dim=1)  # 拼接特征

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return torch.sigmoid(x)  # 输出0-1之间的概率


# --------------------------
# 3. 训练与评估函数
# --------------------------
def train_model(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    model.train()
    best_f1 = 0.0
    best_model_path = "best_qqp_model.pth"

    for epoch in range(epochs):
        train_loss = 0.0
        train_preds = []
        train_labels = []

        # 训练
        for q1, q2, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            q1, q2, label = q1.to(device), q2.to(device), label.to(device)
            optimizer.zero_grad()

            outputs = model(q1, q2).squeeze()
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).float().cpu().numpy())
            train_labels.extend(label.cpu().numpy())

        # 计算训练集指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        dev_loss, dev_acc, dev_f1 = evaluate_model(model, dev_loader, criterion)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"Dev:   Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}")

        # 保存最佳模型（基于验证集F1）
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型（F1={best_f1:.4f}）")

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    return model


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for q1, q2, label in data_loader:
            q1, q2, label = q1.to(device), q2.to(device), label.to(device)
            outputs = model(q1, q2).squeeze()
            loss = criterion(outputs, label)

            total_loss += loss.item()
            preds.extend((outputs > 0.5).float().cpu().numpy())
            labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return avg_loss, acc, f1


# --------------------------
# 4. 主函数
# --------------------------
def main():
    # 配置路径（修改为你的数据集目录）
    data_dir = "D:/pycharm/zuiyouhuashiyan/ruanjiankaifa/QQP"
    batch_size = 64
    epochs = 5
    lr = 0.001

    # 加载数据时增加验证
    result = load_and_preprocess_data(data_dir)
    if result is None:
        print("数据加载失败，程序退出")
        return
    train_dataset, dev_dataset, test_dataset, count_vec, tfidf_transformer = result

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型（输入维度为TF-IDF特征数）
    input_dim = count_vec.max_features  # 5000
    model = QQPClassifier(input_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    print("\n开始训练模型...")
    model = train_model(model, train_loader, dev_loader, criterion, optimizer, epochs=epochs)

    # 测试集评估
    #print("\n在测试集上评估...")
    #test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion)
    #print(f"Test: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}")

    # 预测示例
    def predict_duplicate(question1, question2):
        # 预处理
        q1_counts = count_vec.transform([question1])
        q2_counts = count_vec.transform([question2])
        q1_tfidf = tfidf_transformer.transform(q1_counts)
        q2_tfidf = tfidf_transformer.transform(q2_counts)

        # 转换为张量
        q1_tensor = torch.tensor(q1_tfidf.toarray()[0], dtype=torch.float32).to(device)
        q2_tensor = torch.tensor(q2_tfidf.toarray()[0], dtype=torch.float32).to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            output = model(q1_tensor.unsqueeze(0), q2_tensor.unsqueeze(0))
            return "重复" if output > 0.5 else "不重复"

    # 测试预测功能
    print("\n预测示例:")
    examples = [
        ("What is the best way to learn Python?", "How can I effectively learn Python programming?"),
        ("What is the capital of France?", "How does a computer work?"),
        ("How do I reset my password?", "What's the process to change my password?")
    ]
    for i, (q1, q2) in enumerate(examples, 1):
        print(f"示例 {i}:")
        print(f"问题1: {q1}")
        print(f"问题2: {q2}")
        print(f"预测结果: {predict_duplicate(q1, q2)}\n")


if __name__ == "__main__":
    main()