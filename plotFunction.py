import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
from streamlit_option_menu import option_menu

def printCycle(df,label):
  fig = px.scatter(
    df,
    x='timestamps',
    y="speed(km/h)",
    color=label,
    
  )
  st.plotly_chart(fig)

def energyDisplay(df,labels):
  fig = go.Figure()
  for item in labels:
    fig.add_trace(go.Scatter(
        x=df['timestamps'], y=df[item],
        hoverinfo='x+y',
        mode='lines',
        line=dict(width=0.5),
        stackgroup='one',
        name=item
    ))
  st.plotly_chart(fig)

def energyCompare(df,options):
  traction, thermal, dcdc = [],[],[]
  for id in options:
    piece = df[df['id']==id]
    traction.append(piece['total_traction_power'].max()-piece['total_traction_power'].min())
    thermal.append(piece['total_thermal_power'].max()-piece['total_thermal_power'].min())
    dcdc.append(piece['total_dcdc_power'].max()-piece['total_dcdc_power'].min())


  fig = go.Figure(data=[
      go.Bar(name='Traction Energy', x=options, y=traction),
      go.Bar(name='Thermal Energy', x=options, y=thermal),
      go.Bar(name='DCDC Energy', x=options, y=dcdc)
  ])
  # Change the bar mode
  fig.update_layout(barmode='stack')
  st.plotly_chart(fig)

