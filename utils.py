import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import plotly.graph_objects as go
import plotly.express as px
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

class TreatSkewedVariables():
    def __init__(self, df, threshold):
        self.df = df
        self.threshold = threshold
    
    def YeoJohnson(self):
        numCols = self.df.select_dtypes(include = ['float']).columns.tolist()
        transformer = PowerTransformer(method = 'yeo-johnson')
        transformedCols = []
        for numCol in numCols:
            skewness = self.df[numCol].skew()
            if skewness > self.threshold:
                reshapedCol = self.df[numCol].values.reshape(-1, 1)
                self.df[numCol + '_Unskew'] = transformer.fit_transform(reshapedCol)
        return self.df

class PlotSetup():
    def __init__(self, title, xlab, ylab):
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
    
