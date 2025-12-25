import json

def read_json_config(file_path):
    """
    读取JSON配置文件

    :param file_path:文件路径
    :return: 字典的结果
    """
    try:
        #尝试打开JSON配置文件并加载数据为字典
        with open(file_path, 'r', encoding='utf-8') as json_file:
            config_data = json.load(json_file)
        return config_data
    except FileNotFoundError:
        #若文件未找到，打印错误信息并返回None
        print(f"File not found: {file_path}")
        return None
        #若JSON未找到，打印错误信息并返回None
    except json.JSONDecodeError as e:
        # 如果解析JSON时发生错误，打印错误信息并返回None
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        # 如果发生其他异常，打印错误信息并返回None
        print(f"Happened Exception: {e}")
        return None


# 测试
if __name__ == '__main__':
    data = read_json_config("E:\data\code\config\custom.json")
    print(data)