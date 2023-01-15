import time


def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f'Time taken for method {method.__name__} : {time_taken:.6f} s')
        return result

    return timed
