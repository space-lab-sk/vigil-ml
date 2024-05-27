from datetime import datetime

missions_operational_times = {
    "soho_celias": "1996-01-20 17:30:28",
    "wind_mfi": "1994-11-13 00:00:30"
}

def check_operational_time(mission_instrument: str, start_datetime: str):
    """
    Checks if the provided start datetime is before the operational start datetime of a given mission instrument.

    Args:
        mission_instrument (str): The name or identifier of the mission instrument (e.g. soho_celias, soho_eit195, soho_mag, etc.).
        start_datetime (str):  The datetime to check, formatted as "YYYY-MM-DD HH:MM:SS".

    Raises:
        Exception: If the input start datetime is before the operational start datetime of the specified mission instrument. The exception message includes the operational datetime for reference.
    """

    start_datetime_obj = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")

    operational_datetime_obj = datetime.strptime(missions_operational_times[mission_instrument], "%Y-%m-%d %H:%M:%S")

    if start_datetime_obj < operational_datetime_obj:
        print("start_datetime_obj was before operational_datetime_obj")
        raise Exception(f"Inputed start datetime was before operational start of {mission_instrument}, which was on {missions_operational_times[mission_instrument]}")
    


