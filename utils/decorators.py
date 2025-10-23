import time
from functools import wraps


# 计时装饰器
def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("=" * 30 + f"\n函数{func.__name__}执行时间: {end_time - start_time:.4f}秒\n" + "=" * 30)
        return result
    return wrapper