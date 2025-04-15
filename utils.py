import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import warnings
import seaborn as sns
import polars
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from sklearn.preprocessing import PowerTransformer

### FUNCTIONS


### CLASSES

class DownloadSave():
    def __init__(self, url, path):
        self.url = url
        self.path = path
    def downloadSave(self):
        if os.path.exists(self.path):
            result = pd.read_excel(self.path, engine = 'openpyxl')
            return result
        else:
            try:
                response = requests.get(self.url, stream = True)
                with open(self.path, "wb") as file:
                    for chunk in response.iter_content(chunk_size = 8192):
                        file.write(chunk)
            except requests.exceptions.RequestException as e:
                    print(f"Error {e} occurred while downloading the file.")
            result = pd.read_excel(self.path, engine = "openpyxl")
            return result    

