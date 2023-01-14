from src.utils import str_diff, words2string


class AbstractXyGenerator(object):

    def __init__(self):
        pass

    def gen_rnd_chars(ln: int) -> str:
        return words2string(random.choices(data_dict, k=ln))

    def generate(population_size, xy_data_size):
        pp = []
        for i in range(population_size):
            data = gen_rnd_chars(xy_data_size)
            xy = XY(i, data)
            pp.append(xy)
        return pp
