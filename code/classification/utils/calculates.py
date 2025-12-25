import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


def calculate_eeg_classification_accuracy(outputs, targets):
    # outputs的形状为[batch_size, 2]
    # targets的形状为[batch_size]
    predicted = outputs.argmax(dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100.0
    return accuracy

if __name__ == '__main__':
    # 假设你有一些实际的模型输出和目标标签数据
    outputs = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    targets = torch.tensor([[0, 1], [1, 0], [0, 1]])

    # 调用函数计算准确率
    accuracy = calculate_eeg_classification_accuracy(outputs, targets)

    # 输出准确率
    print(f"Accuracy: {accuracy}")