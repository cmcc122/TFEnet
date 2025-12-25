import logging, os


class SingletonLogger:
    _instance = None

    def __new__(cls, log_file_path):
        if not cls._instance:
            if log_file_path:
                log_file_path = 'logs/train.log'
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._logger = logging.getLogger(__name__)
            cls._instance._logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # 检查并创建文件夹路径
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            cls._instance._logger.addHandler(file_handler)

        return cls._instance

    @staticmethod
    def get_instance(log_file_path:str=None):
        if not SingletonLogger._instance:
            SingletonLogger._instance = SingletonLogger(log_file_path)
        return SingletonLogger._instance

    def log(self, level, message):
        if self._logger:
            self._logger.log(level, message)

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def critical(self, message):
        self.log(logging.CRITICAL, message)


# 示例用法
if __name__ == "__main__":
    log_file_path = "a/example.log"

    # 获取Logger实例
    logger = SingletonLogger.get_instance(log_file_path)

    # 打印日志
    logger.log(logging.INFO, "This is an information message.")
    logger.log(logging.ERROR, "This is an error message.")
