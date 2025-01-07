# #DESCRIPTION:
# Time manager module: to provide some basic operations for time stamp
#
# Created by Hua Yuan, 04/2014
#
# REVISIONS:
# --------------------------------------------------------
import copy
from datetime import datetime, timedelta

daysofmonth_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
daysofmonth_noleap = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
accdaysofmonth_leap = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
accdaysofmonth_noleap = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
LocalLongitude = 0.
isgreenwich = False

class Timestamp:
    def __init__(self, year=0, day=0, sec=0):
        self.year = year
        self.day = day
        self.sec = sec

    def __le__(self, other):
        stsamp1 = copy.copy(self)
        stsamp2 = copy.copy(other)
        adj2end(stsamp1)
        adj2end(stsamp2)

        ts1 = stsamp1.year * 1000 + stsamp1.day
        ts2 = stsamp2.year * 1000 + stsamp2.day

        lessequal = False

        if ts1 < ts2:
            lessequal = True

        if (ts1 == ts2 and stsamp1.sec <= stsamp2.sec):
            lessequal = True
        return lessequal

    def __lt__(self, other):
        stsamp1 = copy.copy(self)
        stsamp2 = copy.copy(other)
        adj2end(stsamp1)
        adj2end(stsamp2)

        ts1 = stsamp1.year * 1000 + stsamp1.day
        ts2 = stsamp2.year * 1000 + stsamp2.day

        lessthan = False

        if ts1 < ts2:
            lessthan = True

        if (ts1 == ts2 and stsamp1.sec < stsamp2.sec):
            lessthan = True
        return lessthan

    def __add__(self, other):
        addsec = Timestamp(self.year, self.day, self.sec)
        addsec.sec += other

        if addsec.sec > 86400:  # More than a day's worth of seconds
            addsec.sec -= 86400
            maxday = 366 if isleapyear(addsec.year) else 365
            addsec.day += 1
            
            if addsec.day > maxday:  # More than a year's worth of days
                addsec.year += 1
                addsec.day = 1

        return addsec

    def __sub__(self, other):
        tstamp1 = Timestamp(self.year, self.day, self.sec)
        tstamp2 = other

        subtstamp = tstamp1 . sec - tstamp2 . sec
        if subtstamp < 0:
            subtstamp = subtstamp + 86400
        return subtstamp



    def subtstamp(self, tstamp1, tstamp2):
        """Subtract the seconds of two timestamps and return the difference."""
        subtstamp = tstamp1.sec - tstamp2.sec
        
        if subtstamp < 0:
            subtstamp += 86400  # Add a day's worth of seconds if the difference is negative
        
        return subtstamp
    


def get_calday(mmdd, isleap):
    """
    Calculate the day of the year given the month and day in mmdd format and whether it's a leap year.

    Args:
        mmdd (int): Month and day in mmdd format (e.g., 123 for January 23).
        isleap (bool): True if it's a leap year, False otherwise.

    Returns:
        int: Day of the year.
    """
    imonth = mmdd // 100
    iday = mmdd % 100

    if isleap:
        return sum(daysofmonth_leap[:imonth]) + iday
    else:
        return sum(daysofmonth_noleap[:imonth]) + iday

def ticktime(deltim, idate):
    idate.sec += int(round(deltim))
    if idate.sec > 86400:
        idate.sec -= 86400
        idate.day += 1

        maxday = 366 if isleapyear(idate.year) else 365

        if idate.day > maxday:
            idate.year += 1
            idate.day = 1
    return idate

def isleapyear(year):
    # Check if the year is a leap year
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def adj2end(idate):
    if idate.sec == 0:
        idate.sec = 86400
        idate.day -= 1
        if idate.day == 0:
            idate.year -= 1
            if isleapyear(idate.year):
                idate.day = 366
            else:
                idate.day = 365
    return idate

def initimetype(greenwich, SinglePoint):
    isgreenwich = greenwich

    if not SinglePoint:
        if not isgreenwich:
            print( 'Warning: Please USE Greenwich time for non-SinglePoint case.')
            isgreenwich = True
    return isgreenwich

def adj2begin(idate):
    if idate.sec == 86400:
        idate.sec = 0
        idate.day = idate.day + 1
        if isleapyear(idate.year) and idate.day == 367:
            idate.year = idate.year + 1
            idate.day = 1

        if not isleapyear(idate.year) and idate.day==366:
            idate.year += 1
            idate.day = 1

    return idate

def minutes_since_1900(year, julianday, second):
    refyear = [1, 1900, 1950, 1980, 1990, 2000, 2005, 2010, 2015, 2020]
    refval = [-998776800, 0, 26297280, 42075360, 47335680, 52594560, 55225440, 
              57854880, 60484320, 63113760]

    iref = max(i for i, y in enumerate(refyear) if y <= year)
    minutes = refval[iref]

    for iyear in range(refyear[iref], year):
        if isleapyear(iyear):
            minutes += 527040
        else:
            minutes += 525600

    minutes += (julianday - 1) * 1440
    minutes += second // 60

    return minutes

def monthday2julian(year, month, mday):
    monthday = None
    if isleapyear(year):
        monthday = accdaysofmonth_leap[:]
    else:
        monthday = accdaysofmonth_noleap[:]

    # calculate julian day
    day = monthday[month - 1] + mday
    return day


def calendarday_date(date, isgreenwich):
    idate = date
    if not isgreenwich:
        idate = localtime2gmt(idate)

    calendarday_date = idate.day + idate.sec / 86400.0
    return calendarday_date

def localtime2gmt(idate):
    tdiff = LocalLongitude / 15.0 * 3600.0
    idate[2] -= int(tdiff)

    if idate[2] < 0:
        idate[2] += 86400
        idate[1] -= 1

        if idate[1] < 1:
            idate[0] -= 1
            if isleapyear(idate[0]):
                idate[1] = 366
            else:
                idate[1] = 365

    if idate[2] > 86400:
        idate[2] -= 86400
        idate[1] += 1

        if isleapyear(idate[0]):
            maxday = 366
        else:
            maxday = 365

        if idate[1] > maxday:
            idate[0] += 1
            idate[1] = 1
    return idate

def julian2monthday(year, day):
    monthday = None
    month = 1

    if isleapyear(year):
        monthday = accdaysofmonth_leap[:]
    else:
        monthday = accdaysofmonth_noleap[:]

    # calculate month and day values
    for i in range(13):
        if day <= monthday[i]:
            month = i
            break
    mday = day - monthday[i-1]
    return month, mday


def isendofhour(idate, sec):
    hour1 = (idate[2]-1)/3600
    hour2 = (idate[2]+int(sec)-1)/3600

    isendofhour = hour1 != hour2
    return isendofhour

def isendofday(idate, sec):
    # Convert idate (array of [year, month, day]) to a datetime object
    tstamp1 = idate
    tstamp2 = tstamp1 + sec
    
    # Check if the day has changed
    if tstamp2.day != tstamp1.day:
        return True
    else:
        return False

def isendofmonth(idate, sec):
    # Convert idate (array of [year, day_of_year]) to a datetime object
    tstamp1 = datetime(idate.year, 1, 1) + timedelta(days=idate.sec - 1)
    
    # Add sec seconds to the timestamp
    tstamp2 = tstamp1 + timedelta(seconds=int(sec))
    
    # Convert Julian days to month and day
    month1, _ = julian2monthday(tstamp1.year, tstamp1.timetuple().tm_yday)
    month2, _ = julian2monthday(tstamp2.year, tstamp2.timetuple().tm_yday)
    
    # Check if the month has changed
    if month1 != month2:
        return True
    else:
        return False
    
def isendofyear(idate, sec):
    # Convert idate (array of [year, month, day]) to a datetime object
    tstamp1 = datetime(idate[0], idate[1], idate[2])
    
    # Add sec seconds to the timestamp
    tstamp2 = tstamp1 + timedelta(seconds=int(sec))
    
    # Check if the year has changed
    if tstamp1.year != tstamp2.year:
        return True
    else:
        return False
# class CoLM_TimeManager(object):
#     def __init__(self, greenwich) -> None:
#         self.isgreenwich = greenwich

        # if not isgreenwich:
        #     print('Warning: Please Use Greenwich time for non-SinglePoint case.')
        #     isgreenwich = True
