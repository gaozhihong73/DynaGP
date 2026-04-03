import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class Logger:
    """
    一个增强的双输出日志器，同时将日志输出到控制台和文件

    主要功能：
    1. 自动创建日志目录
    2. 支持唯一logger名称，避免冲突
    3. 自动文件轮转（可选）
    4. 更完善的错误处理
    5. 支持日志级别动态调整
    6. 线程安全

    使用示例：
        # 基本使用
        logger = Logger("../log/myapp.log")
        logger.info("这是一条信息")

        # 带时间戳的唯一日志文件
        logger = Logger("../log/myapp.log", unique=True)

        # 自定义日志级别
        logger = Logger("../log/myapp.log", file_level='DEBUG', console_level='WARNING')
    """

    _instances = {}  # 类变量，用于存储已创建的logger实例，避免重复创建

    def __init__(self,
                 log_filename: str,
                 file_level: str = 'DEBUG',
                 console_level: str = 'INFO',
                 unique: bool = False,
                 max_bytes: Optional[int] = None,
                 backup_count: int = 5,
                 encoding: str = 'utf-8'):
        """
        初始化日志器

        Args:
            log_filename (str): 日志文件的路径和名称
            file_level (str): 文件日志级别，默认'DEBUG'
            console_level (str): 控制台日志级别，默认'INFO'
            unique (bool): 是否生成唯一的日志文件名（添加时间戳），默认False
            max_bytes (Optional[int]): 单个日志文件最大字节数，None表示不限制
            backup_count (int): 保留的备份文件数量，默认5
            encoding (str): 文件编码，默认'utf-8'
        """
        self.original_filename = log_filename
        self.file_level = self._get_level(file_level)
        self.console_level = self._get_level(console_level)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding

        # 处理唯一文件名
        if unique:
            self.log_filename = self._generate_unique_filename(log_filename)
        else:
            self.log_filename = log_filename

        # 确保日志文件所在目录存在
        self._ensure_log_directory()

        # 生成唯一的logger名称，避免冲突
        self.logger_name = self._generate_logger_name()

        # 检查是否已存在相同配置的logger
        if self.logger_name in Logger._instances:
            existing_logger = Logger._instances[self.logger_name]
            self.logger = existing_logger.logger
            print(f"⚠️  复用现有Logger: {self.logger_name}")
            return

        # 创建新的logger实例
        self._create_logger()

        # 存储实例引用
        Logger._instances[self.logger_name] = self

        print(f"成功创建Logger: {self.log_filename}")

    def _get_level(self, level_str: str) -> int:
        """
        将字符串级别转换为logging级别常量

        Args:
            level_str: 级别字符串

        Returns:
            对应的logging级别常量
        """
        level_mapping = {
            'DEBUG':    logging.DEBUG,
            'INFO':     logging.INFO,
            'WARNING':  logging.WARNING,
            'ERROR':    logging.ERROR,
            'CRITICAL': logging.CRITICAL
            }

        level_str = level_str.upper()
        if level_str not in level_mapping:
            print(f"⚠️  未知的日志级别: {level_str}，使用INFO级别")
            return logging.INFO

        return level_mapping[level_str]

    def _generate_unique_filename(self, original_filename: str) -> str:
        """
        生成带时间戳的唯一文件名

        Args:
            original_filename: 原始文件名

        Returns:
            带时间戳的唯一文件名
        """
        file_path = Path(original_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 在文件名中插入时间戳
        new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        unique_filename = str(file_path.parent / new_name)

        return unique_filename

    def _ensure_log_directory(self) -> None:
        """
        确保日志文件所在目录存在
        """
        try:
            log_dir = os.path.dirname(self.log_filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                print(f"✅ 创建日志目录: {log_dir}")
        except Exception as e:
            print(f"❌ 创建日志目录失败: {e}")
            raise

    def _generate_logger_name(self) -> str:
        """
        生成唯一的logger名称

        Returns:
            唯一的logger名称
        """
        # 使用绝对路径和当前时间生成唯一标识
        abs_path = os.path.abspath(self.log_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"Logger_{abs_path}_{timestamp}_{id(self)}"

    def _create_logger(self) -> None:
        """
        创建logger实例和处理器
        """
        try:
            # 创建logger对象
            self.logger = logging.getLogger(self.logger_name)
            self.logger.setLevel(logging.DEBUG)  # 设置最宽松的级别

            # 清除可能存在的旧handler
            if self.logger.handlers:
                self.logger.handlers.clear()

            # 防止日志传播到根logger（避免重复输出）
            self.logger.propagate = False

            # 创建格式化器
            formatter = self._create_formatter()

            # 创建文件handler
            self._create_file_handler(formatter)

            # 创建控制台handler
            self._create_console_handler(formatter)

        except Exception as e:
            print(f"❌ 创建Logger失败: {e}")
            raise

    def _create_formatter(self) -> logging.Formatter:
        """
        创建日志格式化器

        Returns:
            配置好的格式化器
        """
        return logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )

    def _create_file_handler(self, formatter: logging.Formatter) -> None:
        """
        创建文件处理器

        Args:
            formatter: 日志格式化器
        """
        try:
            if self.max_bytes:
                # 使用轮转文件处理器
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.log_filename,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                    encoding=self.encoding
                    )
            else:
                # 使用普通文件处理器
                file_handler = logging.FileHandler(
                    self.log_filename,
                    encoding=self.encoding,
                    mode='a'  # 追加模式
                    )

            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # 存储file_handler引用以便后续操作
            self.file_handler = file_handler

        except Exception as e:
            print(f"❌ 创建文件处理器失败: {e}")
            raise

    def _create_console_handler(self, formatter: logging.Formatter) -> None:
        """
        创建控制台处理器

        Args:
            formatter: 日志格式化器
        """
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 存储console_handler引用以便后续操作
            self.console_handler = console_handler

        except Exception as e:
            print(f"❌ 创建控制台处理器失败: {e}")
            # 控制台处理器失败不应该阻止程序运行
            pass

    def debug(self, message: str) -> None:
        """记录DEBUG级别信息（通常仅保存到文件）"""
        try:
            self.logger.debug(str(message))
        except Exception as e:
            print(f"❌ 记录debug日志失败: {e}")

    def info(self, message: str) -> None:
        """记录INFO级别信息（同时显示在控制台和文件）"""
        try:
            self.logger.info(str(message))
        except Exception as e:
            print(f"❌ 记录info日志失败: {e}")

    def warning(self, message: str) -> None:
        """记录WARNING级别警告（同时显示在控制台和文件）"""
        try:
            self.logger.warning(str(message))
        except Exception as e:
            print(f"❌ 记录warning日志失败: {e}")

    def error(self, message: str) -> None:
        """记录ERROR级别错误（同时显示在控制台和文件）"""
        try:
            self.logger.error(str(message))
        except Exception as e:
            print(f"❌ 记录error日志失败: {e}")

    def critical(self, message: str) -> None:
        """记录CRITICAL级别严重错误（同时显示在控制台和文件）"""
        try:
            self.logger.critical(str(message))
        except Exception as e:
            print(f"❌ 记录critical日志失败: {e}")

    def set_file_level(self, level: str) -> None:
        """
        动态调整文件日志级别

        Args:
            level: 新的日志级别
        """
        try:
            new_level = self._get_level(level)
            self.file_handler.setLevel(new_level)
            self.file_level = new_level
            self.info(f"文件日志级别已调整为: {level}")
        except Exception as e:
            self.error(f"调整文件日志级别失败: {e}")

    def set_console_level(self, level: str) -> None:
        """
        动态调整控制台日志级别

        Args:
            level: 新的日志级别
        """
        try:
            new_level = self._get_level(level)
            self.console_handler.setLevel(new_level)
            self.console_level = new_level
            self.info(f"控制台日志级别已调整为: {level}")
        except Exception as e:
            self.error(f"调整控制台日志级别失败: {e}")

    def flush(self) -> None:
        """
        强制刷新所有处理器的缓冲区
        """
        try:
            for handler in self.logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            print(f"❌ 刷新日志缓冲区失败: {e}")

    def close(self) -> None:
        """
        关闭日志器，释放资源
        """
        try:
            # 刷新缓冲区
            self.flush()

            # 关闭所有处理器
            for handler in self.logger.handlers[:]:  # 创建副本以避免修改时迭代
                handler.close()
                self.logger.removeHandler(handler)

            # 从实例字典中移除
            if self.logger_name in Logger._instances:
                del Logger._instances[self.logger_name]

            print(f"✅ Logger已关闭: {self.log_filename}")

        except Exception as e:
            print(f"❌ 关闭Logger失败: {e}")

    def get_log_file_path(self) -> str:
        """
        获取当前日志文件的完整路径

        Returns:
            日志文件的绝对路径
        """
        return os.path.abspath(self.log_filename)

    def get_log_file_size(self) -> Union[int, str]:
        """
        获取当前日志文件大小

        Returns:
            文件大小（字节），如果文件不存在返回错误信息
        """
        try:
            if os.path.exists(self.log_filename):
                return os.path.getsize(self.log_filename)
            else:
                return "日志文件不存在"
        except Exception as e:
            return f"获取文件大小失败: {e}"

    def test_logging(self) -> None:
        """
        测试所有日志级别的输出
        """
        print(f"🧪 开始测试Logger: {self.log_filename}")
        print(f"📁 日志文件路径: {self.get_log_file_path()}")

        test_messages = [
            ("DEBUG", "这是一条调试信息"),
            ("INFO", "这是一条普通信息"),
            ("WARNING", "这是一条警告信息"),
            ("ERROR", "这是一条错误信息"),
            ("CRITICAL", "这是一条严重错误信息")
            ]

        for level, message in test_messages:
            method = getattr(self, level.lower())
            method(f"[测试] {message}")

        # 强制刷新
        self.flush()

        # 检查文件是否创建成功
        if os.path.exists(self.log_filename):
            file_size = self.get_log_file_size()
            print(f"✅ 测试完成，日志文件大小: {file_size} 字节")
        else:
            print("❌ 日志文件未创建成功")

    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        try:
            self.close()
        except:
            pass  # 析构时忽略错误

    def __str__(self) -> str:
        return f"Logger(file='{self.log_filename}', file_level={self.file_level}, console_level={self.console_level})"

    def __repr__(self) -> str:
        return self.__str__()


# 便利函数
def create_logger(log_filename: str, **kwargs) -> Logger:
    """
    快速创建Logger的便利函数

    Args:
        log_filename: 日志文件名
        **kwargs: 其他Logger参数

    Returns:
        配置好的Logger实例
    """
    return Logger(log_filename, **kwargs)


def create_timestamped_logger(base_filename: str, **kwargs) -> Logger:
    """
    创建带时间戳的Logger

    Args:
        base_filename: 基础文件名
        **kwargs: 其他Logger参数

    Returns:
        配置好的Logger实例
    """
    return Logger(base_filename, unique=True, **kwargs)


# 使用示例
if __name__ == "__main__":
    print("=== Logger测试 ===")

    # 创建普通logger
    logger1 = Logger("../log/test_app.log")
    logger1.test_logging()

    # 创建带时间戳的unique logger
    logger2 = Logger("../log/unique_app.log", unique=True)
    logger2.test_logging()

    # 创建带文件轮转的logger
    logger3 = Logger("../log/rotating_app.log", max_bytes=1024, backup_count=3)
    logger3.test_logging()

    print("=== 测试完成 ===")
