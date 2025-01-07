import numpy as np


def findloc_ud(array, back=False):
    n = len(array)
    if n <= 0:
        return 0
    else:
        if back:
            i0 = n - 1
            i1 = -1
            ii = -1
        else:
            i0 = 0
            i1 = n
            ii = 1
        result = 0
        for i in range(i0, i1, ii):
            if array[i]:
                result = i
                break
        return result
