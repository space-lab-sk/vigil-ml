from hapiclient import hapi
import numpy as np

def get_hapi_data(server: str, dataset: str, parameters: str, start: str, stop: str) ->np.ndarray:

    opts = {'logging': False} # set True for some more info about transfer
    start, stop = convert_datetime_format(start, stop)

    # TODO: check HAPI availability

    data, meta = hapi(server, dataset, parameters, start, stop, **opts)
    return data


def convert_datetime_format(start: str, stop: str) -> tuple[str, str]:
    """
    Converts a datetime string in the format "YYYY-MM-DD HH:MM:SS" to the format "YYYY-MM-DDTHH:MM:SS", 
    suitable for HAPI

    Args:
        start (str): Starting datetime string to convert.
        stop (str): Stop datetime string to convert.

    Returns:
        The converted datetime string.
    """
    return start.replace(" ", "T"), stop.replace(" ", "T")
