import time
from functools import wraps


def timeit(message_string: str = 'Time taken for {} : {:.6f} s'):
    def decorator(method):
        @wraps(method)
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            time_taken = end_time - start_time
            print('{} {} : {:.6f} s'.format(message_string, method.__name__, time_taken))
            return result

        return timed

    return decorator

# @timeit("Time taken for method")
# def my_method(a, b, c = 3):
#     Method code here
# pass
#
# print(my_method('a', 'b', c = 3))
