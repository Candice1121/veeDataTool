import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
from streamlit_option_menu import option_menu
from asammdf import MDF
from plotFunction import printCycle,energyDisplay,energyCompare 
from computeFunction import findAllZero,deleteZeroDf,speedLabel

with st.sidebar:
    selected = option_menu(
       menu_title='Navigation',
       options=['Processing','Visualization'],
       icons=['house','book']
    )


if selected=='Processing':
  st.title('Electric Vehicle Energy Consumption data tool')
  st.subheader('Input Drving Signal MDF File')
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    st.subheader('Process MDF File to dataframe')
    with st.spinner('Wait for processing...'):
      data = MDF(uploaded_file,raise_on_multiple_occurrences=False)
      dataFiltered = data.to_dataframe(['VehSpdLgtSafe','ALgt1','BkpOfDstTrvld',
                              'HvBattPwr','HvThermPwrCns','RoadIncln','VehM',
                              'HvHeatrPwrCns2','AmbTIndcd'],
                              time_as_date=True)  
      df = dataFiltered.rename(columns={'VehSpdLgtSafe':'speed',
                          'ALgt1':'acceleration',
                          'BkpOfDstTrvld':'total_driven_distance',
                          'HvBattPwr':'output_power',
                          'HvThermPwrCns':'thermal_system_power_consumption',
                          'RoadIncln':'inclination',
                          'VehM':'vehicle_mass',
                          'HvHeatrPwrCns2':'AC_consumption'})
      df['speed(km/h)'] = df['speed'] * 3.6
      df_second = df.resample('0.1S').mean().reset_index().rename(columns={'index':'timestamps'})
      stop = findAllZero(df_second,'speed(km/h)','timestamps')
    st.info('Finished Transformation', icon='\U0001F481')
    if st.button('Show Dataframe'):
      st.dataframe(df_second)
    st.subheader('Delete driving cycle with stop time > 200s (CLTC STANDARD)')
    
    with st.spinner('Wait for deleting...'):
      stop['timeLength'] = [(row['end']-row['start']).total_seconds() for _,row in stop.iterrows()]
      delete = stop.loc[stop['timeLength']>=200]
      idling = stop.loc[stop['timeLength']<200]

      df_200 = deleteZeroDf(df_second,delete)
      df_200['label']='moving'
      for i,r in idling.iterrows():
        label_i = (df_200['timestamps'] >= r['start']) & (df_200['timestamps'] <= r['end'])
        df_200.loc[label_i,'label']='idling'
    st.info('Finished Clearing', icon="\U0001F481")
    if st.button('Show Cleaned Dataframe'):
      st.dataframe(df_200)
    st.subheader('Plot driving cycle in moving and idling')
    with st.spinner('Wait for plotting...'):
      fig = px.scatter(
        df_200,
        x='timestamps',
        y="speed(km/h)",
        color='label'
      )
      st.plotly_chart(fig)
    st.subheader('Generate features for each small driving cycle')
    with st.spinner('Wait for plotting...'):
      label_changes = df_200['label'].ne(df_200['label'].shift())
      group_id = label_changes.cumsum()
      df_200['group_id']=group_id
      df_200['cycle_id']=df_200['group_id']//2
      #plot each cycle with distinct color
      groupNum = df_200['group_id'].max()
      color_labels = df_200['cycle_id'].unique()
      rgb_values = sns.color_palette("Set2", groupNum)
      color_map = dict(zip(color_labels, rgb_values))

      fig, ax = plt.subplots()
      fig.set_figwidth(30)
      Size = 0.3
      for g in np.unique(df_200['cycle_id']):
          ix = np.where(df_200['cycle_id'] == g)

          ax.scatter(df_200['timestamps'].to_numpy()[ix], df_200['speed(km/h)'].to_numpy()[ix], color = color_map[g], label = g, s = Size)

      st.pyplot(fig)
    st.subheader('generate cycle information')
    with st.spinner('Wait for cycle generation...'):
      grouped = df_200.groupby('cycle_id')
      idling_percentage = grouped.apply(lambda x: (x['label'] =='idling').mean()*100)
      speed_avg = grouped['speed(km/h)'].mean()
      speed_max = grouped['speed(km/h)'].max()
      acceleration_avg = grouped['acceleration'].mean()
      acceleration_std = grouped['acceleration'].std()
      mileage = grouped['total_driven_distance'].max() - grouped['total_driven_distance'].min()
      time = grouped['timestamps'].max() - grouped['timestamps'].min()
      energy = grouped['output_power'].sum()

      df_cycle = pd.DataFrame({
          'cycle_id':idling_percentage.index,
          'idling_percentage':idling_percentage.values,
          'average_speed':speed_avg.values,
          'max_speed':speed_max.values,
          'average_acc':acceleration_avg.values,
          'std_acc':acceleration_std.values,
          'mileage':mileage.values,
          'time':time.values,
          'energy':energy.values
      })
      df_cycle['time'] = df_cycle['time'].dt.seconds
      if st.button('Cycle Feature Dataframe'):
        st.write(df_cycle.describe())
    st.subheader('generate speed label')
    for index, item in df_cycle.iterrows():
      df_cycle.loc[index,'speed_label'] = speedLabel(df_cycle.loc[index,'max_speed'])
    df_cycle['cycle_id']=df_cycle['cycle_id'].astype('int64')
    df_200 = df_200.merge(df_cycle[['cycle_id','speed_label']], on='cycle_id',how='left')
    st.write(df_cycle['speed_label'].value_counts())
    st.subheader('plot speed label')
    fig = px.scatter(
      df_200,
      x='timestamps',
      y="speed(km/h)",
      color='speed_label',
      color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig)
  @st.cache_data
  def convert_df(df):
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
      return df.to_csv().encode('utf-8')
  cycle_csv = convert_df(df_cycle)
  st.download_button(
      label="Download cycle dataframe as CSV",
      data=cycle_csv,
      file_name='df_cycle.csv',
      mime='text/csv',
  )

  df_csv = convert_df(df_200)
  st.download_button(
      label="Download driving dataframe as CSV",
      data=cycle_csv,
      file_name='df_driving.csv',
      mime='text/csv',
  )

    





if selected=='Visualization':
  st.title('Driving Cycle Energy Visualization')
  st.subheader('Input Drving CSV')
  uploaded_file = st.file_uploader("Choose a file")

  if uploaded_file is not None:
    ### upload file
    df = pd.read_csv(uploaded_file,parse_dates=[1],index_col=0)
    st.subheader('DataFrame')
    st.write('This driving cycle is from ',df['timestamps'].min(),'to ',df['timestamps'].max())
    df['total_output_power'] = df['output_power'].cumsum()
    df['total_thermal_power'] = df['thermal_power'].cumsum()
    df['total_dcdc_power'] = df['DCDC_power'].cumsum()
    ###label for display
    energyDisplay(df,['total_thermal_power','total_dcdc_power','total_traction_power'])

    
    option = st.selectbox(
      'Choose your label for driving cycle?',
      (df.columns))

    st.write('You selected:', option)

    ### selected driving cycle
    printCycle(df,option)
    drivingID = st.select_slider('select the index of driving cycle: ', range(int(df['id'].max())))
    st.write('For driving cycle',drivingID)
    labels = ['Traction Energy','DCDC Energy','Thermal System Energy']
    traction_power = df['traction_power'].groupby(df['id']).sum()
    dcdc_power = df['DCDC_power'].groupby(df['id']).sum()
    thermal_power = df['thermal_power'].groupby(df['id']).sum()
    energy_values = [traction_power[drivingID],dcdc_power[drivingID],thermal_power[drivingID]]
    movePiece = df[df['id']==drivingID]

    fig = px.line(movePiece['speed(km/h)'])
    st.plotly_chart(fig)

    fig = px.pie(values=energy_values, names=labels)
    st.plotly_chart(fig, use_container_width=True)


    ### group comparison
    
    
    container = st.container()
    all = st.checkbox("Select all")
  
    if all:
        selected_options = container.multiselect('Choose multiple driving cycle to compare',
      range(int(df['id'].max())),range(int(df['id'].max())))
    else:
        selected_options =  container.multiselect(
      'Choose multiple driving cycle to compare',
      range(int(df['id'].max())))

    st.write(selected_options)
    
    energyCompare(df,selected_options)
    #instant percentage

    window = st.select_slider('select the time', df.index)
    st.write('Instant Energy Consumption at at at time:',df.loc[window]['timestamps'])
    values = [df.loc[window,'traction_power'], df.loc[window,'thermal_power'], df.loc[window,'DCDC_power']]
    fig = px.pie(values=values, names=labels)
    st.plotly_chart(fig, use_container_width=True)



    
  else:
    st.info('☝️ Upload a CSV file')



    