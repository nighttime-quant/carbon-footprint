import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os
import warnings
import seaborn as sns
import polars
from scipy import stats
from sklearn.preprocessing import PowerTransformer

### FUNCTIONS

def lineChartFunction(
        fig, df, x, y, title: str, xlab, ylab, 
        theme: str = 'plotly_white', mode:str = 'lines+markers',
        **kwargs
) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode=mode,
            **kwargs
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template=theme
    )
    return fig

def barChartFunction(
        fig, df, x, title:str = None, xlab:str = None, 
        theme:str = 'plotly_white', y = None, ylab:str = None,
        **kwargs
) -> go.Figure:
    fig.add_trace(
        go.Bar(
            x=df[x],
            y=df[y],
            **kwargs
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template=theme
    )
    return fig

def scatterPlotFunction(
        fig, df, x, y, title:str = None, xlab:str = None,
        ylab:str = None, theme:str = 'plotly_white',
        mode:str = 'markers', **kwargs
) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode=mode,
            **kwargs
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template=theme
    )
    return fig

def boxPlotFunction(
        fig, df, y, x = None, title:str = None, ylab:str = None,
        xlab:str = None, theme:str = 'plotly_white', **kwargs
) -> go.Figure:
    fig.add_trace(
        go.Box(
            y=df[y],
            x=df[x],
            **kwargs
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        theme=theme
    )

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
        lineChartFunction(self.fig, self.df, self.xcol, self.ycol, self.title, self.xlab, self.ylab, self.theme)
        return self.fig
    
    def BarChart(self, xcol, title:str = None, xlab:str = None, ycol = None, ylab = None):
        
        self.xcol = xcol
        self.title = title
        self.xlab = xlab or xcol
        self.ycol = ycol
        self.ylab = ylab or ycol
        barChartFunction(self.fig, self.df, self.xcol, self.title, self.xlab, self.theme, self.ycol, self.ylab)
        return self.fig

    def ScatterPlot(self, xcol, ycol, title:str = None, xlab:str = None, ylab:str = None):
        self.xcol = xcol
        self.ycol = ycol
        self.title = title
        self.xlab = xlab or xcol
        self.ylab = ylab or ycol
        scatterPlotFunction(self.fig, self.df, self.xcol, self.ycol, self.title, self.xlab, self.ylab, self.theme)
        return self.fig

    def BoxPlot(self, ycol, xcol = None, ylab:str = None, xlab:str = None, title:str = None):
        self.ycol = ycol
        self.xcol = xcol
        self.ylab = ylab or ycol
        self.xlab = xlab or xcol
        self.title = title
        boxPlotFunction(self.fig, df=self.df, y=self.ycol, x=self.xcol, title=self.title, ylab=self.ylab, xlab=self.xlab, theme=self.theme)
        return self.fig
    
    def show(self):
        self.fig.show()

class DetectOutliers():
    def __init__(self, df):
        self.df = df
        self.z_scores_array = np.array([])
        self.IQR: float = 0.0
        self.lowerQuartile:float = 0.0
        self.upperQuartile:float = 0.0
        self.lowerBound:float = 0.0
        self.upperBound:float = 0.0

    def z_score(self, column, threshold:int = 3, nan_handling = 'omit', **kwargs):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        self.column = column
        self.z_scores_array = stats.zscore(self.df[self.column], nan_policy=nan_handling, **kwargs)
        self.df[self.column + '_z_score'] = self.z_scores_array
        self.df[self.column + '_outlier'] = (self.df[self.column + '_z_score'].abs() > threshold).astype(int)
        return self.df
    
    def iqr(self, column, quartileRange:tuple = (.25, .75), nan_handling = 'omit', interpolation = 'linear', factor:float = 1.5, **kwargs):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        self.column = column
        self.IQR = stats.iqr(self.df[self.column], rng=quartileRange, nan_policy=nan_handling, interpolation=interpolation, **kwargs)
        self.lowerQuartile = self.df[self.column].quantile(quartileRange[0])
        self.upperQuartile = self.df[self.column].quantile(quartileRange[1])
        self.lowerBound = self.lowerQuartile - factor*self.IQR
        self.upperBound = self.upperQuartile + factor*self.IQR
        
        conditions = [
            self.df[self.column] < self.lowerBound,
            self.df[self.column] > self.upperBound
        ]
        choices = [
            1, 1
        ]
        self.df[self.column + '_outlier'] = np.select(conditions, choices, default=0)
        return self.df
