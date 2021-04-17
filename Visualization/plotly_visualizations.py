# Importing libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_histogram(df, text):
    """
    Plotting a subplot histogram for each subcategory
    """

    fig = make_subplots(
        rows = 2, cols = 5,
        subplot_titles = (tuple(df.columns))
    )
    rows_idx = 1
    cols_idx = 1
    for col in df.columns:
        fig.add_trace(go.Histogram(x = df.loc[:, col]), row = rows_idx, col = cols_idx)
        if cols_idx == 5:
            rows_idx += 1
            cols_idx = 1
        else:
            cols_idx += 1
    fig.update_layout(height=900, width=900, title_text=text)
    fig.show()

def create_box_plot(df, text):
    """
    Plotting a subplot boxplot for each subcategory
    """

    fig = make_subplots(
        rows = 2, cols = 5,
        subplot_titles = (tuple(df.columns))
    )
    rows_idx = 1
    cols_idx = 1
    for col in df.columns:
        fig.add_trace(go.Box(x = df.loc[:, col]), row = rows_idx, col = cols_idx)
        if cols_idx == 5:
            rows_idx += 1
            cols_idx = 1
        else:
            cols_idx += 1
    fig.update_layout(height=900, width=900, title_text=text)
    fig.show()
    

def creat_violin_plot(df, text):
    """
    Plots violin plot for each sub category
    """

    fig = make_subplots(
        rows = 2, cols = 5,
        subplot_titles = (tuple(df.columns))
    )
    rows_idx = 1
    cols_idx = 1
    for col in df.columns:
        fig.add_trace(go.Violin(x = df.loc[:, col], meanline_visible = True), row = rows_idx, col = cols_idx)
        if cols_idx == 5:
            rows_idx += 1
            cols_idx = 1
        else:
            cols_idx += 1
    fig.update_layout(height=900, width=900, title_text=text)
    fig.show()

def main():
    """
    Divide the dataset into Mean, SE and Worst and visualize different attributes of the data
    """

    df = pd.read_csv('../Dataset/Breast Cancer Data.csv')
    # Drop 'id' and 'Unnamed:32'
    df = df.drop('id', axis=1)
    df = df.drop('Unnamed: 32', axis=1)

    # Grouping the datat as _mean, _se, _worst
    df_mean = df.iloc[:, 1:11]
    df_se = df.iloc[:, 11:21]
    df_worst = df.iloc[:, 21:]

    # Histogram Plotting
    create_histogram(df_mean, 'Histogram of the Mean Data')
    create_histogram(df_se, 'Histogram of the SE Data')
    create_histogram(df_worst, 'Histogram of the Worst Data')

    # Box Plotting
    create_box_plot(df_mean, 'Box Plot of the Mean Data')
    create_box_plot(df_se, 'Box Plot of the SE Data')
    create_box_plot(df_worst, 'Box Plot of the Worst Data')

    # Violin Plots
    creat_violin_plot(df_mean, 'Violion Plot of Mean Data')
    creat_violin_plot(df_se, 'Violion Plot of SE Data')
    creat_violin_plot(df_worse, 'Violion Plot of Worst Data')

    # Heatmap
    fig = px.imshow(df.corr(), title='HeatMap')
    fig.show()
    

if __name__ == '__main__':
    main()