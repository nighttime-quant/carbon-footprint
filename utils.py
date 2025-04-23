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

def lineChartFunction(df, x, y, title: str, xlab, ylab, theme: str = 'plotly_white'):
    fig = px.line(df, x = x, y = y, title = title, labels = {x: xlab, y: ylab}, template=theme)
    return fig

def barChartFunction(df, x, title: str, xlab: str, theme: str = 'plotly_white', y = None, ylab: str = None):
    fig = px.bar(df, x = x, y = y, title = title, labels = {x: xlab, y: ylab}, template=theme)
    return fig

def scatterPlotFunction(df, x, y, title:str, xlab:str = None, ylab:str = None, theme: str = 'plotly_white'):
    fig = px.scatter(data_frame=df, x=x, y=y, labels={x: xlab, y: ylab}, template=theme)
    return fig

class PlotSetup():
    def __init__(
            self, 
            df, 
            width:int = 800,
            height:int = 600,
            theme: str = 'plotly_white'
        ):
        

        self.df = df
        self.theme = theme
        self.width = width
        self.height = height        

        self.fig = go.Figure()

        self.fig.update_layout(
            template = theme,
            width = width,
            height = height
        )

class DataVisualizer(PlotSetup):
    def __init__(self, df, width:int = 800, height:int = 600, theme:str = 'plotly_white'):
        super().__init__(df = df, width = width, height = height, theme = theme)
    
    def LineChart(self, xcol, ycol, title:str = None, xlab:str = None, ylab:str = None):
        
        self.xcol = xcol
        self.ycol = ycol
        self.title = title
        self.xlab = xlab or xcol
        self.ylab = ylab or ycol
        self.fig = lineChartFunction(self.df, self.xcol, self.ycol, self.title, self.xlab, self.ylab, self.theme)
        return self.fig
    
    def BarChart(self, xcol, title:str, xlab:str, ycol = None, ylab = None):
        
        self.xcol = xcol
        self.title = title
        self.xlab = xlab or xcol
        self.ycol = ycol
        self.ylab = ylab or ycol
        self.fig = barChartFunction(self.df, self.xcol, self.title, self.xlab, self.theme, self.ycol, self.ylab)

    def ScatterPlot(self, xcol, ycol, title:str = None, xlab:str = None, ylab:str = None):
        self.xcol = xcol
        self.ycol = ycol
        self.title = title
        self.xlab = xlab or xcol
        self.ylab = ylab or ycol
        self.fig = scatterPlotFunction(self.df, self.xcol, self.ycol, self.title, self.xlab, self.ylab, self.theme)

    def show(self):
        self.fig.show()