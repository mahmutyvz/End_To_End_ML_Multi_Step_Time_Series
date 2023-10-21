import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def date_column_info(data, streamlit=False):
    """
    This function generates line charts for numeric columns in a DataFrame based on a datetime column, providing insights into the data's temporal trends.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the datetime column and numeric columns for visualization.
    streamlit (bool, optional): If True, the function uses Streamlit to display the charts. If False (default), the function uses Plotly.
    Returns:

    None
    """
    num_cols = data.select_dtypes(
        include=['float', 'int']).columns.tolist()
    fig = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols)

    for i, col in enumerate(num_cols):
        line_chart = px.line(data, x='date', y=col,template='plotly_dark')
        line = line_chart.data[0]
        fig.add_trace(line, row=i + 1, col=1)
    num_rows = data.shape[1]
    fig.update_xaxes(title_text='Date', row=num_cols, col=1)
    fig.update_layout(showlegend=False, height=150 * num_rows, width=1400)
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)


def pred_visualize(y_val, y_pred, col_index, external_or_real, streamlit=False):
    """
    This function visualizes predicted and actual values for a specific column in a time series.

    Parameters:

    y_val (pandas.Series): The actual values for the target column.
    y_pred (pandas.Series): The predicted values for the target column.
    col_index (int): The index of the column to visualize.
    external_or_real (str): A label indicating whether the values are external or real.
    streamlit (bool, optional): If True, the function uses Streamlit to display the chart. If False (default), the function uses Plotly.
    Returns:

    None
    """
    fig = go.Figure()
    y_val_vis = y_val.iloc[:, col_index].astype('float32')
    y_pred_vis = y_pred.iloc[:, col_index].astype('float32')
    real_table_name = y_val_vis.name.split('lag_')[1]
    fig.add_trace(
        go.Scatter(x=y_val.index, y=y_val_vis, mode='lines', name=f'{external_or_real} Real Day {real_table_name}',
                   line_color='#247AFD'))
    fig.add_trace(
        go.Scatter(x=y_val.index, y=y_pred_vis, mode='lines', name=f'{external_or_real} Pred Day {real_table_name}',
                   line_color='#ff0000'))
    fig.update_layout(title=f'{external_or_real} Test Değerleri ve Tahminler',
                      xaxis_title='Time',
                      yaxis_title='Horizon Time Steps')
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
