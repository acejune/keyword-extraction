import logging

from extractor_config import TOP_N


def set_logger(logger_name):
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # formatter 지정
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)7s] [%(filename)18s:%(lineno)4d] (%(funcName)20s) >> %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        save_path = f"_{logger_name}.log"
        file_handler = logging.FileHandler(filename=save_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def sort_keyword(keyword_set):
    keyword_dict = {}
    for keyword, score in keyword_set:
        before_socre = keyword_dict.get(keyword, -99)
        if before_socre < score:
            keyword_dict[keyword] = round(score, 5)
    keyword_list = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)

    return keyword_list[:TOP_N]
