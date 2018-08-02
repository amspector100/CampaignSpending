import glob
import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class XML2DF():
    """Converts XML file to pandas dataframe. Adapted from http://austintaylor.io/lxml/python/pandas/xml/dataframe/2016/07/08/convert-xml-to-pandas-dataframe/."""

    def __init__(self, xml_path):
        # Read text and parse as XML
        self.root = ET.parse(xml_path)

    # Get relevant information
    def find_data(self):
        income = self.root.find('income')
        return income


def process_directory(directory_path):
    glob_pattern = directory_path + '\*.xml*'
    paths = glob.glob(glob_pattern)
    for path in tqdm(paths):
        # Takes about 30 seconds for q1 2016
        c = XML2DF(path)
        income = c.find_data()
    all_tags = set()






path2 = "data/raw/lobbying_disclosures/Q12016/300784631.xml"
process_directory('data/raw/lobbying_disclosures/Q12016')
xml = XML2DF(path2)

import sys
sys.exit()
path = "data/raw/lobbying_disclosures/2004_Registrations_XML/8042078.xml"
print(XML2DF(path).process_data())


