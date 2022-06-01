import logging
import sys


class Logger:
    @classmethod
    def getLogger(cls, name: str, store_in_file=False) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = (
            logging.FileHandler("app.log")
            if store_in_file
            else logging.StreamHandler(sys.stdout)
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
