from functools import reduce


class Utils:
    @staticmethod
    def pipeline(x, *funcs):
        """
        Applies a sequence of methods to a single argument.
        """
        return reduce(lambda x, f: f(x), funcs, x)
