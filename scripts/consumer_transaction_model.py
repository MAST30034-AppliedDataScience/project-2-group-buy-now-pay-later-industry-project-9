import holidays



def is_holiday(date):
    """
    Get all holidays in Australia in 2021
    """
    country = 'AU'
    year = 2021
    holiday_calendar = holidays.CountryHoliday(country, years=[year])
    return date in holiday_calendar
