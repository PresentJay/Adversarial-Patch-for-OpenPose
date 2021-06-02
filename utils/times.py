import time


def get_current_time():
    now = time.localtime()
    return f'{now.tm_year}{now.tm_mon}-{now.tm_mday}_{now.tm_hour}-{now.tm_min}-{now.tm_sec}'