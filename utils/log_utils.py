import logging
import six
import os
import sys


class LogIdFilter(logging.Filter):
    def filter(self, record):
        logid = getattr(record, "tags", {}).get("_logid")
        if logid:
            record._logid = logid
            return True
        logid = os.getenv('OPERATOR_LOG_ID')
        record._logid = six.ensure_text(logid) if logid is not None else '-'
        return True


def setup_logger(tag: str) -> logging.Logger:
    
    logger = logging.getLogger(tag)
    # 如果logger已经有handlers，说明已经被配置过，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    # 同时给 root 也放一个 stdout handler，兜底其它模块的日志
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        root.addHandler(logging.StreamHandler(sys.stdout))
    
    # 阻止日志向上传播到root logger，避免重复输出
    logger.propagate = False
    
    return logger


logger = setup_logger("ttlive_agent")
