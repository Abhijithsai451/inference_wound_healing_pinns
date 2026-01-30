import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(
        name= "inference_wound_healing_pinns",
        log_file="app_log.log",
        level=logging.DEBUG,
        max_bytes=10*1024*1024,
        backup_count=5
    ):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s')

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
