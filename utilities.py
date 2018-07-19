import numpy as np
import pandas as pd
import csv
import time

# Data processing utilities ----------------------------------------------
def get_filelength(path, encoding = 'Latin-1', **kwargs):
    """
    Checks the length of a file.
    :param path: Path of file, presumably a csv
    :param kwargs: kwargs to pass to open(filepath, 'r') call.
    :return:
    """
    with open(path, 'r', encoding = encoding, **kwargs) as file:
        row_count = sum(1 for row in file) # Use generator for efficiency
    return row_count