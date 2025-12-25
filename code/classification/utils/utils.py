import json
import os

import torch


def is_cuda(judge):
    return torch.device("cuda" if judge else "cpu")


def save_model_checkpoint(model, checkpoint_path):
    """
    保存模型权重的函数

    Parameters:
    - model: 模型实例
    - checkpoint_path (str): 保存检查点的文件路径
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved at: {checkpoint_path}')


def load_model_checkpoint(model, checkpoint_path):
    """
    读取模型权重的函数

    Parameters:
    - model: 模型实例
    - checkpoint_path (str): 检查点文件路径
    """
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Model weights loaded from: {checkpoint_path}')


def save_dict_as_json(dictionary, output_path, file_name):
    """
    将字典保存为JSON文件

    Parameters:
        dictionary (dict): 要保存的字典
        output_path (str): 文件输出路径
        file_name (str): 要保存的文件名
    """
    # 构建完整的输出文件路径
    full_path = os.path.join(output_path, file_name)
    # 如果输出路径不存在，则创建
    os.makedirs(output_path, exist_ok=True)
    # 将字典保存为JSON文件
    with open(full_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

