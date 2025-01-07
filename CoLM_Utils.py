import math
import numpy as np
from decimal import Decimal


def overload_decorator(*conditions):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = None
            for condition, num_parameter in conditions:
                # print(*args)
                if len(args) == num_parameter:
                    result = condition(*args, **kwargs)
                # if condition(*args, **kwargs):
                #     return inner_func(*args, **kwargs)
            return func(result)
        return wrapper
    return decorator

def normalize_longitude(lon):
    while lon >= 180.0:
        lon = lon - 360.0

    while lon < -180.0:
        lon = lon + 360.0
    return lon


def find_nearest_south(y, n, lat):
    iloc = 0

    if lat[0] < lat[n - 1]:
        if y <= lat[0]:
            iloc = 1
        elif y >= lat[n - 1]:
            iloc = n
        else:
            ileft = 1
            iright = n

            while iright - ileft > 1:
                i = (iright + ileft) // 2
                if y >= lat[i - 1]:
                    ileft = i
                else:
                    iright = i
            iloc = ileft
    else:
        if y >= lat[0]:
            iloc = 1
        elif y <= lat[n - 1]:
            iloc = n
        else:
            ileft = 1
            iright = n

            while iright - ileft > 1:
                i = (iright + ileft) // 2
                if y >= lat[i - 1]:
                    iright = i
                else:
                    ileft = i

            iloc = iright

    return iloc - 1


def find_nearest_north(y, n, lat):
    iloc = 0

    if lat[0] < lat[n - 1]:
        if y <= lat[0]:
            iloc = 1
        elif y >= lat[n - 1]:
            iloc = n
        else:
            ileft = 1
            iright = n

            while iright - ileft > 1:
                i = (iright + ileft) // 2
                if y > lat[i - 1]:
                    ileft = i
                else:
                    iright = i
            iloc = iright
    else:
        if y >= lat[0]:
            iloc = 1
        elif y <= lat[n - 1]:
            iloc = n
        else:
            ileft = 1
            iright = n

            while iright - ileft > 1:
                i = (iright + ileft) // 2
                if y > lat[i - 1]:
                    iright = i
                else:
                    ileft = i

            iloc = ileft
    return iloc - 1


def find_nearest_west(x, n, lon):
    iloc = 0

    if n == 1:
        return 0

    if lon_between_floor(x, lon[n - 1], lon[0]):
        return n - 1

    ileft = 1
    iright = n
    while iright - ileft > 1:
        i = (iright + ileft) // 2
        if lon_between_floor(x, lon[i - 1], lon[iright - 1]):
            ileft = i
        else:
            iright = i

    iloc = ileft
    return iloc - 1


def find_nearest_east(x, n, lon):
    iloc = 0

    if n == 1:
        return 0

    if lon_between_ceil(x, lon[n - 1], lon[0]):
        return 0

    ileft = 1
    iright = n
    while iright - ileft > 1:
        i = (iright + ileft) // 2
        if lon_between_ceil(x, lon[i - 1], lon[iright - 1]):
            ileft = i
        else:
            iright = i

    iloc = iright
    return iloc - 1


def lon_between_floor(lon, west, east):
    # lon, west, east[-180, 180)
    lon_between_floor = False

    if west >= east:
        lon_between_floor = (lon >= west) or (lon < east)
    else:
        lon_between_floor = (lon >= west) and (lon < east)
    return lon_between_floor


def lon_between_ceil(lon, west, east):
    # lon, west, east [-180, 180)
    lon_between_ceil = False

    if west >= east:
        lon_between_ceil = (lon > west) or (lon <= east)
    else:
        lon_between_ceil = (lon > west) and (lon <= east)
    return lon_between_ceil


def median(x, spval=None):
    xtemp = None
    if spval is not None:
        msk = x != spval
        nc = len(np.where(msk)[0])

        if nc != 0:
            xtemp = x[msk]
            del msk

        else:
            mval = spval
            del msk
            return mval
    else:
        xtemp = x
        nc = len(x)

    if nc % 2 == 0:
        left = quickselect(nc, xtemp, nc // 2)
        right = quickselect(nc, xtemp, nc // 2 + 1)
        mval = (left + right) / 2.0
    else:
        mval = quickselect(nc, xtemp, nc // 2 + 1)
    del xtemp

    return mval


def quickselect(nA, A, k):
    # A = np.zeros(nA)
    if nA > 0:
        pivot = A[(nA-1) // 2]
        left = -1
        right = nA

        while left < right:
            right -= 1
            while A[right] > pivot:
                right -= 1

            left += 1
            while A[left] < pivot:
                left += 1

            if left < right:
                A[left], A[right] = A[right], A[left]

        marker = right

        if k <= marker + 1:
            return quickselect(marker , A[:marker+1], k)
        else:
            return quickselect(nA - marker-1, A[marker + 1:], k - marker-1)
    else:
        return A[0]


def insert_into_sorted_list(x, n, list):
    if n == 0:
        iloc = 1
        is_new = True
    elif x <= list[0]:
        iloc = 1
        is_new = (x != list[0])
    elif x > list[n - 1]:
        iloc = n + 1
        is_new = True
    elif x == list[n - 1]:
        iloc = n
        is_new = False
    else:
        ileft = 1
        iright = n

        while True:
            if iright - ileft > 1:
                iloc = (ileft + iright) // 2
                if x > list[iloc - 1]:
                    ileft = iloc
                elif x < list[iloc - 1]:
                    iright = iloc
                else:
                    is_new = False
                    break
            else:
                iloc = iright
                is_new = True
                break

    if is_new:
        if iloc <= n:
            list[iloc:n+1] = list[iloc-1:n]
        list[iloc -1] = x
        n += 1

    return iloc - 1, is_new, n, list


def expand_list(data, percent):
    n0 = len(data)
    temp = data
    n1 = int(np.ceil(n0 * (1 + percent)))

    del data
    out = np.zeros(n1, dtype=int)
    out[0:n0] = temp
    del temp
    return out

def arclen(lat1, lon1, lat2, lon2):
    """
    Calculate the arc length (great-circle distance) between two points on the Earth's surface.
    
    Parameters:
    lat1, lon1 (float): Latitude and longitude of the first point (in radians).
    lat2, lon2 (float): Latitude and longitude of the second point (in radians).
    
    Returns:
    float: The arc length in kilometers.
    """
    
    re = 6371.22  # Earth's radius in kilometers

    return re * math.acos(math.sin(lat1) * math.sin(lat2) +
                          math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2))

def find_in_sorted_list(x, n, lst):
    iloc = 0
    if n > 0:
        if x >= lst[0] and x <= lst[n - 1]:
            if x == lst[0]:
                iloc = 0
            elif x == lst[n - 1]:
                iloc = n - 1
            else:
                ileft = 0
                iright = n - 1
                while iright - ileft > 0:
                    i = (ileft + iright) // 2
                    if x == lst[i]:
                        iloc = i
                        break
                    elif x > lst[i]:
                        ileft = i
                    elif x < lst[i]:
                        iright = i
    return iloc


def areaquad(lats, latn, lonw, lone):
    re = 6371.22  # Earth radius in kilometers
    # deg2rad = np.pi / 180.0  # Conversion factor from degrees to radians
    deg2rad = 1.745329251994330e-2
    if lone < lonw:
        dx = (lone + 360 - lonw) * deg2rad
    else:
        dx = (lone - lonw) * deg2rad

    dy = np.sin(latn * deg2rad) - np.sin(lats * deg2rad)

    area = dx * dy * re * re  # Area calculation in square kilometers

    return area

def quicksort(nA, A, order):

    if nA > 1:
        medianindex = nA // 2
        pivot = A[medianindex-1]
        left = -1
        right = nA

        while left < right:
            right = right - 1
            while A[right] > pivot:
                right = right - 1

            left = left + 1
            while A[left] < pivot:
                left = left + 1

            if left < right:
                temp = A[left]
                A[left] = A[right]
                A[right] = temp

                temp = order[left]
                order[left] = order[right]
                order[right] = temp

        marker = right


        if len(A[0:marker+1])>1:
            A[0:marker + 1], order[0:marker + 1] = quicksort(marker +1, A[0:marker+1],order[0:marker+1])
            # A[0:marker + 1], order[0:marker + 1] = temp1, temp2
        if len(A[marker+1:nA]) > 1:
            A[marker+1:nA], order[marker+1:nA] = quicksort(nA-marker-1, A[marker+1:nA],order[marker+1:nA])
            # A[marker:nA], order[marker:nA] = temp3, temp4

    # if nA > 1:
    #     medianindex = nA // 2
    #     pivot = A[medianindex-1]
    #     left = -1
    #     right = nA
    #
    #     while left < right:
    #         right = right - 1
    #         while A[right] > pivot:
    #             right = right - 1
    #
    #         left = left + 1
    #         while A[left] < pivot:
    #             left = left + 1
    #
    #         if left < right:
    #             temp = A[left]
    #             A[left] = A[right]
    #             A[right] = temp
    #
    #             temp = order[left]
    #             order[left] = order[right]
    #             order[right] = temp
    #
    #     marker = right
    #
    #     temp1, temp2 = quicksort(marker + 1, A[0:marker + 1], order[0:marker + 1])
    #     A[0:marker + 1], order[0:marker + 1] = temp1, temp2
    #     temp3, temp4 = quicksort(nA - marker - 1, A[marker:nA], order[marker:nA])
    #     A[marker:nA], order[marker:nA] = temp3, temp4
        return A, order



def insert_into_sorted_list2(x, y, n, xlist, ylist):
    ileft = 0
    iright = n - 1
    is_new = False

    if n == 0:
        iloc = 0
        is_new = True
    elif y < ylist[0] or (y == ylist[0] and x <= xlist[0]):
        iloc = 0
        is_new = (x != xlist[0]) or (y != ylist[0])
    elif y > ylist[iright] or (y == ylist[iright] and x > xlist[iright]):
        iloc = n
        is_new = True
    elif x == xlist[iright] and y == ylist[iright]:
        iloc = n -1
    else:
        ileft = 0
        iright = n - 1
        while True:
            if iright - ileft > 1:
                iloc = (ileft + iright) // 2
                if y > ylist[iloc] or (y == ylist[iloc] and x > xlist[iloc]):
                    ileft = iloc
                elif y < ylist[iloc] or (y == ylist[iloc] and x < xlist[iloc]):
                    iright = iloc
                else:
                    is_new = False
                    break
            else:
                iloc = iright
                is_new = True
                break

    if is_new:
        if iloc < n:
            xlist[iloc + 1:n + 1] = xlist[iloc:n]
            ylist[iloc + 1:n + 1] = ylist[iloc:n]

        xlist[iloc] = x
        ylist[iloc] = y
        n += 1

    return n, xlist, ylist, iloc, is_new


def insert_into_sorted_list1(x, n, lst):
    ileft = 0
    iright = n - 1
    is_new = False

    if n == 0:
        iloc = 0
        is_new = True
    elif x <= lst[0]:
        iloc = 0
        is_new = (x != lst[0])
    elif x > lst[iright]:
        iloc = n
        is_new = True
    elif x == lst[iright]:
        iloc = iright
    else:
        while iright - ileft > 1:
            iloc = (ileft + iright) // 2
            if x > lst[iloc]:
                ileft = iloc
            elif x < lst[iloc]:
                iright = iloc
            else:
                is_new = False
                break
        else:
            iloc = iright
            is_new = True

    if is_new:
        if iloc < n:
            lst[iloc + 1:n + 1] = lst[iloc:n]

        lst[iloc] = x
        n += 1

    return n, lst, iloc, is_new


def find_in_sorted_list2(x, y, n, xlist, ylist):
    iloc = 0
    if n < 1:
        return 0

    if y < ylist[0] or (y == ylist[0] and x < xlist[0]):
        return 0
    elif y > ylist[n - 1] or (y == ylist[n - 1] and x > xlist[n - 1]):
        return 0
    elif x == xlist[0] and y == ylist[0]:
        return 1
    elif x == xlist[n - 1] and y == ylist[n - 1]:
        return n
    else:
        ileft = 0
        iright = n - 1

        while True:
            if iright - ileft > 1:
                i = (ileft + iright) // 2
                if y == ylist[i] and x == xlist[i]:
                    iloc = i + 1
                    break
                elif y > ylist[i] or (y == ylist[i] and x > xlist[i]):
                    ileft = i
                elif y < ylist[i] or (y == ylist[i] and x < xlist[i]):
                    iright = i
            else:
                iloc = 0
                break

    return iloc

def polint(xa, ya, n, x):
    """
    Given arrays xa and ya, each of length n, and given a value x, this routine
    returns a value y and an error estimate dy. If P(x) is the polynomial of
    degree N-1 such that P(xa(i)) = ya(i), i = 1, . . . , n, then the returned
    value y = P(x).
    """

    # Largest anticipated value
    NMAX = 10
    c = [0.0] * NMAX
    d = [0.0] * NMAX

    ns = 0
    dif = abs(x - xa[0])

    for i in range(n):
        dift = abs(x - xa[i])
        if dift < dif:
            ns = i
            dif = dift
        c[i] = ya[i]
        d[i] = ya[i]

    y = ya[ns]
    ns -= 1

    for m in range(n - 1):
        for i in range(n - m - 1):
            ho = xa[i] - x
            hp = xa[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            if den == 0.0:
                print('failure in polint')  # two input xa's are identical.
            den = w / den
            d[i] = hp * den
            c[i] = ho * den

        if 2 * ns < n - m:
            dy = c[ns + 1]
        else:
            dy = d[ns]
            ns -= 1
        y += dy

    return y


def subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s, isiter, L_vgm, nprint, infos=None):
    info = 0
    if infos is not None:
        info = infos
    if iflag < 0:
        info = iflag

    iflag = 0

    if 0 < nprint:
        if L_vgm is not None:
            fvec, fjac, isiter = sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                    isiter, L_vgm)
        else:
            fvec, fjac, isiter = sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                    isiter)

    return info, fvec, fjac, isiter


def sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydatv, ydatvks, nptf, phi, k_s, isiter, L_vgm):
    if iflag == 0:
        print(x)
    elif iflag == 1:
        if x[1] <= 0.0 or x[2] <= 0.1 or x[2] >= 1000.0 or x[3] <= 0.0:
            isiter = 0
            return fvec, fjac

        for i in range(m):
            temp = []
            for j in ydatvks[:, i]:
                temp.append(math.log10(j))
            fvec[i] = (sum(
                ((x[0] + (phi - x[0]) * (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1) - ydatv[:, i]) / phi) ** 2) +
                       sum(((math.log10(x[3]) + (1.0 / x[2] - 1) * L_vgm * math.log10(1 + (x[1] * xdat[i]) ** x[2]) +
                             math.log10(
                                 (1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (
                                         1 - 1.0 / x[2])) ** 2) - temp) / math.log10(k_s)) ** 2))

    elif iflag == 2:
        if x[1] <= 0.0 or x[2] <= 0.1 or x[2] >= 1000.0 or x[3] <= 0.0:
            isiter = 0
            return fvec, fjac, isiter

        for i in range(m):
            temp = []
            for j in ydatvks[:, i]:
                temp.append(math.log10(j))
            fjac[i, 0] = sum(
                2 * ((x[0] + (phi - x[0]) * (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1) - ydatv[:, i]) / phi) *
                (1 - (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1)) / phi)

            fjac[i, 1] = sum(2 * ((x[0] + (phi - x[0]) * (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1) -
                                   ydatv[:, i]) / phi) / phi * (phi - x[0]) * (1 - x[2]) * (
                                     1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 2) * x[1] ** (
                                     x[2] - 1) * xdat[i] ** x[2]) + sum(
                2 * ((math.log10(x[3]) + (1.0 / x[2] - 1) * L_vgm * math.log10(1 + (x[1] * xdat[i]) ** x[2]) +
                      math.log10((1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (
                              1 - 1.0 / x[2])) ** 2) - temp) / math.log10(k_s)) *
                (L_vgm * (1.0 - x[2]) * x[1] ** (x[2] - 1) * xdat[i] ** x[2] / (
                        (1 + (x[1] * xdat[i]) ** x[2]) * math.log(10.)) + 2.0 * (1.0 - x[2]) * (
                         1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (-1.0 / x[2]) *
                 x[1] ** (x[2] - 1) * xdat[i] ** x[2] * (1 + (x[1] * xdat[i]) ** x[2]) ** (-2) /
                 ((1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (1 - 1.0 / x[2])) * math.log(
                     10.))) / math.log10(k_s))

            fjac[i, 2] = sum(2 * ((x[0] + (phi - x[0]) * (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1) -
                                   ydatv[:, i]) / phi) / phi *
                             (phi - x[0]) * (1 + (x[1] * xdat[i]) ** x[2]) ** (1.0 / x[2] - 1) *
                             ((1.0 - x[2]) * (x[1] * xdat[i]) ** x[2] * math.log(x[1] * xdat[i]) / (
                                     x[2] * (1 + (x[1] * xdat[i]) ** x[2])) -
                              math.log(1 + (x[1] * xdat[i]) ** x[2]) / x[2] ** 2)) + sum(
                2 * ((math.log10(x[3]) + (1.0 / x[2] - 1) * L_vgm * math.log10(1 + (x[1] * xdat[i]) ** x[2]) +
                      math.log10((1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (
                              1 - 1.0 / x[2])) ** 2) - temp) / math.log10(k_s)) *
                (-1.0 * L_vgm * math.log10(1 + (x[1] * xdat[i]) ** x[2]) / x[2] ** 2 +
                 (1.0 / x[2] - 1) * L_vgm * (x[1] * xdat[i]) ** x[2] * math.log10(x[1] * xdat[i]) / (
                         1 + (x[1] * xdat[i]) ** x[2]) -
                 2.0 * (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (1 - 1.0 / x[2]) /
                 (1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** (1 - 1.0 / x[2])) *
                 (math.log10(1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) / x[2] ** 2 +
                  (1 - 1.0 / x[2]) * math.log10(x[1] * xdat[i]) / (1 + (x[1] * xdat[i]) ** x[2]))) / math.log10(
                    k_s))

            fjac[i, 3] = sum(
                2 * ((math.log10(x[3]) + (1.0 / x[2] - 1) * L_vgm * math.log10(1 + (x[1] * xdat[i]) ** x[2]) +
                      math.log10((1.0 - (1.0 - 1.0 / (1 + (x[1] * xdat[i]) ** x[2])) ** ( 1-1.0/x[3]))**2)-
                        temp)/math.log10(k_s))/ (x[3] * math.log(10.)) / math.log10(k_s))

    return fvec, fjac, isiter


def sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydatc, ydatcks, nptf, phi, k_s, isiter):
    if iflag == 0:
        print(x)
    elif iflag == 1:
        if x[0] >= 0.0 or abs(x[1]) >= 100.0 or x[2] <= 0.0:
            isiter = 0
            return fvec, fjac, isiter
        for i in range(m):
            fvec[i] = np.sum((((-1.0 * xdat[i] / x[0]) ** (-1.0 * x[1]) * phi - ydatc[:, i]) / phi) ** 2) + np.sum(
                ((np.log10(-1.0 * xdat[i] / x[0]) * (-3.0 * x[1] - 2) + np.log10(x[2]) - np.log10(
                    ydatcks[:, i])) / np.log10(k_s)) ** 2)
    elif iflag == 2:
        if x[0] >= 0.0 or abs(x[1]) >= 100.0 or x[2] <= 0.0:
            isiter = 0
            return fvec, fjac
        for i in range(m):
            fjac[i, 0] = np.sum(2.0 * (((-1.0 * xdat[i] / x[0]) ** (-1.0 * x[1]) * phi - ydatc[:, i]) / phi) *
                                x[1] * (-1.0 * xdat[i] / x[0]) ** (-1.0 * x[1]) / x[0]) + np.sum(
                2.0 * (((np.log10(-1.0 * xdat[i] / x[0]) * (-3.0 * x[1] - 2) + np.log10(
                    x[2]) - np.log10(ydatcks[:, i])) / np.log10(k_s))) *
                (3.0 * x[1] + 2) / (x[0] * np.log(10.)) / np.log10(k_s))
            fjac[i, 1] = np.sum(-2.0 * (((-1.0 * xdat[i] / x[0]) ** (-1.0 * x[1]) * phi - ydatc[:, i]) / phi) *
                                (-1.0 * xdat[i] / x[0]) ** (-1.0 * x[1]) * np.log(-1.0 * xdat[i] / x[0])) + np.sum(
                -6.0 * (((np.log10(-1.0 * xdat[i] / x[0]) * (-3.0 * x[1] - 2) + np.log10(
                    x[2]) - np.log10(ydatcks[:, i])) / np.log10(k_s))) *
                np.log10(-1.0 * xdat[i] / x[0]) / np.log10(k_s))
            fjac[i, 2] = np.sum(2.0 * (((np.log10(-1.0 * xdat[i] / x[0]) * (-3.0 * x[1] - 2) + np.log10(
                x[2]) - np.log10(ydatcks[:, i])) / np.log10(k_s))) /
                                (x[2] * np.log(10.) * np.log10(k_s)))
    return fvec, fjac, isiter


# *****************************************************************************
#
# ! QRFAC computes a QR factorization using Householder transformations.
#
#  Discussion:
#
#    This FUNCTION uses Householder transformations with optional column
#    pivoting to compute a QR factorization of the
#    M by N matrix A.  That is, QRFAC determines an orthogonal
#    matrix Q, a permutation matrix P, and an upper trapezoidal
#    matrix R with diagonal elements of nonincreasing magnitude,
#    such that A*P = Q*R.
#
#    The Householder transformation for column K, K = 1,2,...,min(M,N),
#    is of the form
#
#      I - ( 1 / U(K) ) * U * U'
#
#    WHERE U has zeros in the first K-1 positions.
#
#    The form of this transformation and the method of pivoting first
#    appeared in the corresponding LINPACK routine.
#
#  Licensing:
#
#    This code may freely be copied, modified, and used for any purpose.
#
#  Modified:
#
#    06 April 2010
#
#  Author:
#
#    Original FORTRAN77 version by Jorge More, Burton Garbow, Kenneth Hillstrom.
#    FORTRAN90 version by John Burkardt.
#
#  Reference:
#
#    Jorge More, Burton Garbow, Kenneth Hillstrom,
#    User Guide for MINPACK-1,
#    Technical Report ANL-80-74,
#    Argonne National Laboratory, 1980.
#
#  Parameters:
#
#    Input, integer ( kind = 4 ) M, the number of rows of A.
#
#    Input, integer ( kind = 4 ) N, the number of columns of A.
#
#    Input/output, real ( kind = 8 ) A(LDA,N), the M by N array.
#    On input, A CONTAINS the matrix for which the QR factorization is to
#    be computed.  On output, the strict upper trapezoidal part of A CONTAINS
#    the strict upper trapezoidal part of R, and the lower trapezoidal
#    part of A CONTAINS a factored form of Q, the non-trivial elements of
#    the U vectors described above.
#
#    Input, integer ( kind = 4 ) LDA, the leading dimension of A, which must
#    be no less than M.
#
#    Input, logical PIVOT, is TRUE IF column pivoting is to be carried out.
#
#    Output, integer ( kind = 4 ) IPVT(LIPVT), defines the permutation matrix P
#    such that A*P = Q*R.  Column J of P is column IPVT(J) of the identity
#    matrix.  IF PIVOT is false, IPVT is not referenced.
#
#    Input, integer ( kind = 4 ) LIPVT, the dimension of IPVT, which should
#    be N IF pivoting is used.
#
#    Output, real ( kind = 8 ) RDIAG(N), CONTAINS the diagonal elements of R.
#
#    Output, real ( kind = 8 ) ACNORM(N), the norms of the corresponding
#    columns of the input matrix A.  IF this information is not needed,
#    THEN ACNORM can coincide with RDIAG.
def qrfac(m, n, a, lda, pivot, ipvt, lipvt, rdiag, acnorm):
    wa = np.zeros(n)
    epsmch = np.finfo(float).eps

    for j in range(n):
        acnorm[j] = np.linalg.norm(a[:m, j])

    rdiag[:n - 1] = acnorm[:n - 1]
    wa[:n - 1] = acnorm[:n - 1]

    if pivot:
        for j in range(n):
            ipvt[j] = j

    minmn = min(m, n)

    for j in range(minmn):
        if pivot:
            kmax = j
            for k in range(j, n):
                if rdiag[kmax] < rdiag[k]:
                    kmax = k

            if kmax != j:
                r8_temp = a[:m - 1, j].copy()
                a[:m - 1, j] = a[:m - 1, kmax]
                a[:m - 1, kmax] = r8_temp

                rdiag[kmax] = rdiag[j]
                wa[kmax] = wa[j]

                i4_temp = ipvt[j]
                ipvt[j] = ipvt[kmax]
                ipvt[kmax] = i4_temp

        ajnorm = np.linalg.norm(a[j, j])

        if ajnorm != 0.0:
            if a[j, j] < 0.0:
                ajnorm = -ajnorm

            a[j:m - 1, j] /= ajnorm
            a[j, j] += 1.0

            for k in range(j + 1, n):
                temp = np.dot(a[j:m - 1, j], a[j:m - 1, k]) / a[j, j]
                a[j:m - 1, k] -= temp * a[j:m - 1, j]

                if pivot and rdiag[k] != 0.0:
                    temp = a[j, k] / rdiag[k]
                    rdiag[k] *= np.sqrt(max(0.0, 1.0 - temp ** 2))

                    if 0.05 * (rdiag[k] / wa[k]) ** 2 <= epsmch:
                        rdiag[k] = np.linalg.norm(a[j + 1:, k])
                        wa[k] = rdiag[k]

        rdiag[j] = -ajnorm
    return a, ipvt, rdiag, acnorm


# *****************************************************************************
#
# ! QRSOLV solves a rectangular linear system A*x=b in the least squares sense.
#
#  Discussion:
#
#    Given an M by N matrix A, an N by N diagonal matrix D,
#    and an M-vector B, the problem is to determine an X which
#    solves the system
#
#      A*X = B
#      D*X = 0
#
#    in the least squares sense.
#
#    This FUNCTION completes the solution of the problem
#    IF it is provided with the necessary information from the
#    QR factorization, with column pivoting, of A.  That is, IF
#    A*P = Q*R, WHERE P is a permutation matrix, Q has orthogonal
#    columns, and R is an upper triangular matrix with diagonal
#    elements of nonincreasing magnitude, THEN QRSOLV expects
#    the full upper triangle of R, the permutation matrix p,
#    and the first N components of Q'*B.
#
#    The system is THEN equivalent to
#
#      R*Z = Q'*B
#      P'*D*P*Z = 0
#
#    WHERE X = P*Z.  IF this system does not have full rank,
#    THEN a least squares solution is obtained.  On output QRSOLV
#    also provides an upper triangular matrix S such that
#
#      P'*(A'*A + D*D)*P = S'*S.
#
#    S is computed within QRSOLV and may be of separate interest.
#
#  Licensing:
#
#    This code may freely be copied, modified, and used for any purpose.
#
#  Modified:
#
#    06 April 2010
#
#  Author:
#
#    Original FORTRAN77 version by Jorge More, Burton Garbow, Kenneth Hillstrom.
#    FORTRAN90 version by John Burkardt.
#
#  Reference:
#
#    Jorge More, Burton Garbow, Kenneth Hillstrom,
#    User Guide for MINPACK-1,
#    Technical Report ANL-80-74,
#    Argonne National Laboratory, 1980.
#
#  Parameters:
#
#    Input, integer ( kind = 4 ) N, the order of R.
#
#    Input/output, real ( kind = 8 ) R(LDR,N), the N by N matrix.
#    On input the full upper triangle must contain the full upper triangle
#    of the matrix R.  On output the full upper triangle is unaltered, and
#    the strict lower triangle CONTAINS the strict upper triangle
#    (transposed) of the upper triangular matrix S.
#
#    Input, integer ( kind = 4 ) LDR, the leading dimension of R, which must be
#    at least N.
#
#    Input, integer ( kind = 4 ) IPVT(N), defines the permutation matrix P such
#    that A*P = Q*R.  Column J of P is column IPVT(J) of the identity matrix.
#
#    Input, real ( kind = 8 ) DIAG(N), the diagonal elements of the matrix D.
#
#    Input, real ( kind = 8 ) QTB(N), the first N elements of the vector Q'*B.
#
#    Output, real ( kind = 8 ) X(N), the least squares solution.
#
#    Output, real ( kind = 8 ) SDIAG(N), the diagonal elements of the upper
#    triangular matrix S.
def qrsolv(n, r, ipvt, diag, qtb, x, sdiag):
    wa = np.zeros(n, dtype=float)

    for j in range(n):
        r[j:n - 1, j] = r[j, j:n - 1]
        x[j] = r[j, j]

    wa[:n - 1] = qtb[:n - 1]

    for j in range(n):
        l = ipvt[j]
        if diag[l] != 0.0:
            sdiag[j:n] = 0.0
            sdiag[j] = diag[l]
            qtbpj = 0.0

            for k in range(j, n):
                if sdiag[k] != 0.0:
                    if abs(r[k, k]) < abs(sdiag[k]):
                        cotan = r[k, k] / sdiag[k]
                        s = 0.5 / np.sqrt(0.25 + 0.25 * cotan ** 2)
                        c = s * cotan
                    else:
                        t = sdiag[k] / r[k, k]
                        c = 0.5 / np.sqrt(0.25 + 0.25 * t ** 2)
                        s = c * t

                    r[k, k] = c * r[k, k] + s * sdiag[k]
                    temp = c * wa[k] + s * qtbpj
                    qtbpj = -s * wa[k] + c * qtbpj
                    wa[k] = temp

                    for i in range(k + 1, n):
                        temp = c * r[i, k] + s * sdiag[i]
                        sdiag[i] = -s * r[i, k] + c * sdiag[i]
                        r[i, k] = temp

            sdiag[j] = r[j, j]
            r[j, j] = x[j]

    nsing = n

    for j in range(n):
        if sdiag[j] == 0.0 and nsing == n:
            nsing = j - 1
        if nsing < n:
            wa[j] = 0.0
    # print(nsing,'*********nsing*******')
    for j in range(nsing - 1 , 0, -1):
        sum2 = np.dot(wa[j + 1:nsing], r[j + 1:nsing, j])
        wa[j] = (wa[j] - sum2) / sdiag[j]

    for j in range(n):
        l = ipvt[j]
        x[l] = wa[j]

    return x, sdiag


def lmpar(n, r, ldr, ipvt, diag, qtb, delta, par, x, sdiag):
    dwarf = np.finfo(float).tiny

    nsing = n
    par = 0

    wa1 = np.zeros(n)
    wa2 = np.zeros(n)
    for j in range(n):
        wa1[j] = qtb[j]
        if r[j, j] == 0.0 and nsing == n:
            nsing = j - 1
        if nsing < n:
            wa1[j] = 0.0

    for k in range(nsing):
        j = nsing - k - 1
        wa1[j] = wa1[j] / r[j, j]
        temp = wa1[j]
        wa1[:j] -= r[:j, j] * temp

    for j in range(n):
        l = ipvt[j]
        x[l] = wa1[j]

    iter = 0
    wa2[0:n - 1] = diag[0:n - 1] * x[0:n - 1]
    dxnorm = np.linalg.norm(wa2)

    fp = dxnorm - delta

    if fp <= 0.1 * delta:
        if iter == 0:
            par = 0.0
        return par, x, sdiag

    parl = 0.0

    if n <= nsing:
        wa1 = diag * (wa2 / dxnorm)
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)

        for j in range(n):
            sum2 = np.dot(wa1[:j], r[:j, j])
            wa1[j] = (wa1[j] - sum2) / r[j, j]
        temp = np.linalg.norm(wa1)
        parl = (fp / delta / temp) / temp

    for j in range(n):
        sum2 = np.dot(qtb[:j], r[:j, j])
        l = ipvt[j]
        wa1[j] = sum2 / diag[l]

    gnorm = np.linalg.norm(wa1)
    paru = gnorm / delta

    if paru == 0.0:
        paru = dwarf / min(delta, 0.1)

    par = max(par, parl)
    par = min(par, paru)
    if par == 0.0:
        par = gnorm / dxnorm

    while True:
        iter += 1

        if par == 0.0:
            par = max(dwarf, 0.001 * paru)

        wa1[0:n - 1] = np.sqrt(par) * diag[0:n - 1]
        x, sdiag = qrsolv(n, r, ipvt, wa1, qtb, x, sdiag)

        wa2[0:n - 1] = diag[0:n - 1] * x[0:n - 1]
        dxnorm = np.linalg.norm(wa2)
        temp = fp
        fp = dxnorm - delta

        if abs(fp) <= 0.1 * delta:
            break

        if parl == 0.0 and fp <= temp < 0.0:
            break
        elif iter == 10:
            break

        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[j] * (wa2[j] / dxnorm)

        for j in range(n):
            wa1[j] = wa1[j] / sdiag[j]
            temp = wa1[j]
            wa1[j + 1:n] -= r[j + 1:n, j] * temp

        temp = np.linalg.norm(wa1)
        parc = (fp / delta / temp) / temp

        if 0.0 < fp:
            parl = max(parl, par)
        elif fp < 0.0:
            paru = min(paru, par)

        par = max(parl, par + parc)

    if iter == 0:
        par = 0.0

    return par, x, sdiag


# *******************************************************************************
#
# ! LMDER minimizes M functions in N variables by the Levenberg-Marquardt method
#  implemented for fitting the SW retention & hydraulic conductivity parameters
#  in the Campbell/van Genuchten models.
#
#  Discussion:
#
#    LMDER minimizes the sum of the squares of M nonlinear functions in
#    N variables by a modification of the Levenberg-Marquardt algorithm.
#    The user must provide a subroutine which calculates the functions
#    and the jacobian.
#
#  Licensing:
#
#    This code may freely be copied, modified, and used for any purpose.
#
#  Modified:
#
#    06 April 2010
#
#  Author:
#
#    Original FORTRAN77 version by Jorge More, Burton Garbow, Kenneth Hillstrom.
#    FORTRAN90 version by John Burkardt.
#    Modified by Nan Wei, 2019/01
#
#  Reference:
#
#    Jorge More, Burton Garbow, Kenneth Hillstrom,
#    User Guide for MINPACK-1,
#    Technical Report ANL-80-74,
#    Argonne National Laboratory, 1980.
#
#  Parameters:
#
#    Input, external FCN, the name of the user-supplied subroutine which
#    calculates the functions and the jacobian.  FCN should have the form:
#      subroutine fcn ( m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s, isiter, L_vgm)
#      integer ( kind = 4 ) ldfjac
#      integer ( kind = 4 ) n
#      real ( kind = 8 ) fjac(ldfjac,n)
#      real ( kind = 8 ) fvec(m)
#      integer ( kind = 4 ) iflag
#      real ( kind = 8 ) x(n)
#      xdat, npoint, ydat, ydatks, nptf, phi, k_s and isiter are transfered as the inputs of the fitting functions.
#      L_vgm are only used for vanGenuchten_Mualem soil model input.
#
#    If IFLAG = 0 on input, then FCN is only being called to allow the user
#    to print out the current iterate.
#    If IFLAG = 1 on input, FCN should calculate the functions at X and
#    return this vector in FVEC.
#    If IFLAG = 2 on input, FCN should calculate the jacobian at X and
#    return this matrix in FJAC.
#    To terminate the algorithm, FCN may set IFLAG negative on return.
#
#    Input, integer ( kind = 4 ) M, is the number of functions.
#
#    Input, integer ( kind = 4 ) N, is the number of variables.
#    N must not exceed M.
#
#    Input/output, real ( kind = 8 ) X(N).  On input, X must contain an initial
#    estimate of the solution vector.  On output X contains the final
#    estimate of the solution vector.
#
#    Output, real ( kind = 8 ) FVEC(M), the functions evaluated at the output X.
#
#    Output, real ( kind = 8 ) FJAC(LDFJAC,N), an M by N array.  The upper
#    N by N submatrix of FJAC contains an upper triangular matrix R with
#    diagonal elements of nonincreasing magnitude such that
#      P' * ( JAC' * JAC ) * P = R' * R,
#    where P is a permutation matrix and JAC is the final calculated jacobian.
#    Column J of P is column IPVT(J) of the identity matrix.  The lower
#    trapezoidal part of FJAC contains information generated during
#    the computation of R.
#
#    Input, integer ( kind = 4 ) LDFJAC, the leading dimension of FJAC.
#    LDFJAC must be at least M.
#
#    Input, real ( kind = 8 ) FTOL.  Termination occurs when both the actual
#    and predicted relative reductions in the sum of squares are at most FTOL.
#    Therefore, FTOL measures the relative error desired in the sum of
#    squares.  FTOL should be nonnegative.
#
#    Input, real ( kind = 8 ) XTOL.  Termination occurs when the relative error
#    between two consecutive iterates is at most XTOL.  XTOL should be
#    nonnegative.
#
#    Input, real ( kind = 8 ) GTOL.  Termination occurs when the cosine of the
#    angle between FVEC and any column of the jacobian is at most GTOL in
#    absolute value.  Therefore, GTOL measures the orthogonality desired
#    between the function vector and the columns of the jacobian.  GTOL should
#    be nonnegative.
#
#    Input, integer ( kind = 4 ) MAXFEV.  Termination occurs when the number of
#    calls to FCN with IFLAG = 1 is at least MAXFEV by the end of an iteration.
#
#    Input/output, real ( kind = 8 ) DIAG(N).  If MODE = 1, then DIAG is set
#    internally.  If MODE = 2, then DIAG must contain positive entries that
#    serve as multiplicative scale factors for the variables.
#
#    Input, integer ( kind = 4 ) MODE, scaling option.
#    1, variables will be scaled internally.
#    2, scaling is specified by the input DIAG vector.
#
#    Input, real ( kind = 8 ) FACTOR, determines the initial step bound.  This
#    bound is set to the product of FACTOR and the euclidean norm of DIAG*X if
#    nonzero, or else to FACTOR itself.  In most cases, FACTOR should lie
#    in the interval (0.1, 100) with 100 the recommended value.
#
#    Input, integer ( kind = 4 ) NPRINT, enables controlled printing of iterates
#    if it is positive.  In this case, FCN is called with IFLAG = 0 at the
#    beginning of the first iteration and every NPRINT iterations thereafter
#    and immediately prior to return, with X and FVEC available
#    for printing.  If NPRINT is not positive, no special calls
#    of FCN with IFLAG = 0 are made.
#
#    Output, integer ( kind = 4 ) INFO, error flag.  If the user has terminated
#    execution, INFO is set to the (negative) value of IFLAG. See description
#    of FCN.  Otherwise, INFO is set as follows:
#    0, improper input parameters.
#    1, both actual and predicted relative reductions in the sum of
#       squares are at most FTOL.
#    2, relative error between two consecutive iterates is at most XTOL.
#    3, conditions for INFO = 1 and INFO = 2 both hold.
#    4, the cosine of the angle between FVEC and any column of the jacobian
#       is at most GTOL in absolute value.
#    5, number of calls to FCN with IFLAG = 1 has reached MAXFEV.
#    6, FTOL is too small.  No further reduction in the sum of squares
#       is possible.
#    7, XTOL is too small.  No further improvement in the approximate
#       solution X is possible.
#    8, GTOL is too small.  FVEC is orthogonal to the columns of the
#       jacobian to machine precision.
#
#    Output, integer ( kind = 4 ) NFEV, the number of calls to FCN with
#    IFLAG = 1.
#
#    Output, integer ( kind = 4 ) NJEV, the number of calls to FCN with
#    IFLAG = 2.
#
#    Output, integer ( kind = 4 ) IPVT(N), defines a permutation matrix P
#    such that JAC*P = Q*R, where JAC is the final calculated jacobian, Q is
#    orthogonal (not stored), and R is upper triangular with diagonal
#    elements of nonincreasing magnitude.  Column J of P is column
#    IPVT(J) of the identity matrix.
#
#    Output, real ( kind = 8 ) QTF(N), contains the first N elements of Q'*FVEC.
def lmder(m, n, x, fvec, fjac, ldfjac, ftol, xtol, gtol, maxfev,
     mode, factor, nprint, xdat, npoint, ydat,
     ydatks, nptf, phi, k_s, isiter, L_vgm=None):
    epsmch = np.finfo(float).eps

    info = 0
    iflag = 0
    nfev = 0
    njev = 0
    ipvt = None
    qtf = None
    diag = np.zeros(n)
    delta = 0
    temp = 0
    xnorm = 0

    if n <= 0 or m < n or ldfjac < m or ftol < 0.0 or xtol < 0.0 or gtol < 0.0 or maxfev <= 0 or factor <= 0.0:
        info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                      isiter, L_vgm, nprint)
        return x, isiter

    if mode == 2 and any(diag <= 0.0):
        info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                      isiter, L_vgm, nprint)
        return x, isiter

    iflag = 1
    if L_vgm is not None:
        fvec, fjac,isiter = sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s, isiter,
                                L_vgm)
    else:
        fvec, fjac,isiter = sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s, isiter)
    nfev = 1
    if iflag < 0:
        info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                      isiter, L_vgm, nprint)
        return x, isiter

    fnorm = np.linalg.norm(fvec)

    par = 0.0
    iter = 1

    while True:
        iflag = 2
        if L_vgm is not None:
            fvec, fjac, isiter = sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                    isiter, L_vgm)
        else:
            fvec, fjac, isiter = sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                    isiter)
        njev += 1

        if iflag < 0:
            info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                          k_s, isiter, L_vgm,
                                          nprint)
            return x, isiter

        if 0 < nprint:
            if iter % nprint == 0:
                iflag = 0
                if L_vgm is not None:
                    fvec, fjac = sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                            k_s, isiter,
                                            L_vgm)

                else:
                    fvec, fjac, isiter = sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                            k_s, isiter)

                if iflag < 0:
                    info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf,
                                                  phi, k_s, isiter,L_vgm, nprint)
                    return x, isiter

        pivot = True
        qtf = np.zeros(n)
        ipvt = np.arange(n)
        wa1 = np.zeros(n)
        wa2 = np.zeros(n)
        wa3 = np.zeros(n)
        wa4 = np.zeros(m)
        fjac, ipvt, wa1, wa2 = qrfac(m, n, fjac, ldfjac, pivot, ipvt, n, wa1, wa2)
        if iter == 1:
            if mode != 2:
                diag[0:n - 1] = wa2[0:n - 1]
                for j in range(n):
                    if wa2[j] == 0:
                        diag[j] = 1
            wa3[0: n - 1] = diag[0: n - 1] * x[0: n - 1]
            xnorm = np.linalg.norm(wa3)
            if xnorm == 0:
                delta = factor
            else:
                delta = factor * xnorm
        wa4[0: m - 1] = fvec[0: m - 1]

        for j in range(n):
            if fjac[j, j] != 0.0:
                sum2 = np.dot(wa4[j:m - 1], fjac[j:m - 1, j])
                temp = -sum2 / fjac[j, j]
                wa4[j:m - 1] += fjac[j:m - 1, j] * temp
            fjac[j, j] = wa1[j]
            qtf[j] = wa4[j]

        gnorm = 0.0
        if fnorm != 0.0:
            for j in range(n):
                l = ipvt[j]
                if wa2[l] != 0.0:
                    sum2 = np.dot(qtf[0:j], fjac[0:j, j]) / fnorm
                    gnorm = max(gnorm, abs(sum2 / wa2[l]))

        if gnorm <= gtol:
            info = 4
            info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                          k_s, isiter, L_vgm,
                                          nprint, info)
            return x, isiter

        if mode != 2:
            for j in range(n):
                diag[j] = np.maximum(diag[j], wa2[j])

        while True:
            par, wa1, wa2 = lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1, wa2)
            wa1[0:n] = -wa1[0:n]
            wa2[0:n] = x[0:n] + wa1[0:n]
            wa3[0:n] = np.diag(np.arange(1, n + 1)) @ wa1[0:n]

            pnorm = np.linalg.norm(wa3)

            if iter == 1:
                delta = min(delta, pnorm)

            iflag = 1
            if L_vgm is not None:
                fvec, fjac, isiter = sw_vg_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                        isiter, L_vgm)
            else:
                fvec, fjac, isiter = sw_cb_dist(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi, k_s,
                                        isiter)
            nfev += 1

            if iflag < 0:
                info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                              k_s, isiter,
                                              L_vgm, nprint)
                return x, isiter

            fnorm1 = np.linalg.norm(wa4)

            if 0.1 * fnorm1 < fnorm:
                actred = 1.0 - (fnorm1 / fnorm) ** 2
            else:
                actred = -1.0

            for j in range(n):
                wa3[j] = 0.0
                l = ipvt[j]
                temp = wa1[l]
                wa3[0:j + 1] += fjac[0:j + 1, j] * temp

            temp1 = np.linalg.norm(wa3) / fnorm
            temp2 = (np.sqrt(par) * pnorm) / fnorm
            prered = temp1 ** 2 + temp2 ** 2 / 0.5
            dirder = -(temp1 ** 2 + temp2 ** 2)

            if prered != 0.0:
                ratio = actred / prered
            else:
                ratio = 0.0

            if ratio <= 0.25:
                if 0.0 <= actred:
                    temp = 0.5
                if actred < 0.0:
                    temp = 0.5 * dirder / (dirder + 0.5 * actred)
                if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                    temp = 0.1
                delta = temp * min(delta, pnorm / 0.1)
                par = par / temp
            else:
                if par == 0.0 or ratio >= 0.75:
                    delta = 2.0 * pnorm
                    par = 0.5 * par

            if 0.0001 <= ratio:
                x[0:n - 1] = wa2[0:n - 1]
                wa2[0:n - 1] = np.diag(np.arange(0, n - 1)) @ x[0:n - 1]
                fvec[0:m - 1] = wa4[0:m - 1]
                xnorm = np.linalg.norm(wa2)
                fnorm = fnorm1
                iter += 1

            if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0:
                info = 1

            if delta <= xtol * xnorm:
                info = 2

            if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0 and info == 2:
                info = 3

            if info != 0:
                info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                              k_s, isiter,
                                              L_vgm, nprint,info)
                return x, isiter

            if nfev >= maxfev:
                info = 5

            if abs(actred) <= epsmch and prered <= epsmch and 0.5 * ratio <= 1.0:
                info = 6

            if delta <= epsmch * xnorm:
                info = 7

            if gnorm <= epsmch:
                info = 8

            if info != 0:
                info, fvec, fjac, isiter = subprocess(m, n, x, fvec, fjac, ldfjac, iflag, xdat, npoint, ydat, ydatks, nptf, phi,
                                              k_s, isiter,
                                              L_vgm, nprint,info)
                return x, isiter

            if 0.0001 <= ratio:
                break

    # return x

def append_to_list(list1, list2):
    list1.extend(list2)
    # n1 = 0
    # if list1 is not None:
    #     n1=len(list1)
    #
    # n2 = len(list2)
    #
    # if n1 > 0:
    #     temp = list1.copy()
    #     list1 = np.zeros(n1+n2)
    #     list1[0:n1-1] = temp
    #     del temp
    # else:
    #     if  n2 > 0:
    #         list1 = np.zeros(n2)
    #
    # if  n1 + n2 > 0:
    #     list1[n1 : n1 + n2 -1] = list2
    return list1

def tridia(n, a, b, c, r, u):
    """
    Solves the tridiagonal system of equations.
    
    Parameters:
    n (int): Length of the diagonal element vector
    a (array-like): Subdiagonal elements
    b (array-like): Diagonal elements
    c (array-like): Superdiagonal elements
    r (array-like): Right hand side
    
    Returns:
    array-like: Solution vector
    """
    gam = np.zeros(n)
    
    bet = b[0]
    u[0] = Decimal(str(r[0])) / Decimal(str(bet))
    
    for j in range(1, n):
        gam[j] = Decimal(str(c[j-1])) / Decimal(str(bet))
        bet = Decimal(str(b[j])) - Decimal(str(a[j])) * Decimal(str(gam[j]))
        u[j] = (Decimal(str(r[j])) - Decimal(str(a[j])) * Decimal(str(u[j-1]))) / Decimal(str(bet))
    
    for j in range(n-2, -1, -1):
        u[j] = Decimal(str(u[j]))-Decimal(str(gam[j+1])) * Decimal(str(u[j+1]))
    
    return u
