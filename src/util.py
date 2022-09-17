from datetime import datetime

def getDayOfWeek(date):
    # Must be a string in YYYY-MM-DD
    date_object = datetime.strptime(date, "%Y\%m\%d")
    return date_object.isoweekday()
