import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from PIL import Image
import datetime

st.set_page_config(
    layout = 'wide'
)

st.markdown('### **Reference Panel Ancestry**')

pie1, pie2 = st.columns(2)

out_path = f'data/GP2_QC_round3_MDGAP-QSBB'
ref_pca_path = f'{out_path}_labeled_ref_pca.txt'
ref_pca = pd.read_csv(ref_pca_path, sep='\s+')

df_ancestry_counts = ref_pca['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
ref_counts = ref_pca['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
ref_combo = pd.merge(df_ancestry_counts, ref_counts, on='Ancestry Category')

pie2.dataframe(ref_combo)

pie_chart = px.pie(df_ancestry_counts, names = 'Ancestry Category', values = 'Proportion', hover_name = 'Ancestry Category',  color="Ancestry Category", 
            color_discrete_map={'AFR':'#8a0c9a',
                                 'SAS':'#01a6aa',
                                 'EAS':'#14b472',
                                 'EUR':'#43591b',
                                 'AMR': '#2b56b7',
                                 'AJ': '#ff8df6',
                                 'AAC': '#f2c219',
                                 'CAS': '#927459',
                                 'MDE': '#fc7000',
                                 'FIN': '#a11505'})
pie_chart.update_layout(showlegend = True, width=500,height=500)
pie1.plotly_chart(pie_chart)


st.markdown('### **Predicted Ancestry**')

pie3, pie4 = st.columns(2)

df_new_counts = st.session_state.combined['Predicted Ancestry'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
new_counts = st.session_state.combined['Predicted Ancestry'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
new_combo = pd.merge(df_new_counts, new_counts, on='Ancestry Category')

pie4.dataframe(new_combo)

pie_chart = px.pie(df_new_counts, names = 'Ancestry Category', values = 'Proportion', hover_name = 'Ancestry Category',  color="Ancestry Category", 
            color_discrete_map={'AFR':'#8a0c9a',
                                 'SAS':'#01a6aa',
                                 'EAS':'#14b472',
                                 'EUR':'#43591b',
                                 'AMR': '#2b56b7',
                                 'AJ': '#ff8df6',
                                 'AAC': '#f2c219',
                                 'CAS': '#927459',
                                 'MDE': '#fc7000',
                                 'FIN': '#a11505'})
pie_chart.update_layout(showlegend = True, width=500,height=500)
pie3.plotly_chart(pie_chart)

st.markdown('### **Reference Panel vs. Predicted Ancestry Counts**')

bar1, bar2 = st.columns(2)

bar_compare = go.Figure(
            data=[
                go.Bar(y=ref_counts['Ancestry Category'], x=ref_counts['Counts'], orientation='h', name="Reference Panel", base=0, marker_color='rgb(55, 83, 109)'),
                go.Bar(y=new_counts['Ancestry Category'], x=-new_counts['Counts'], orientation='h', name="Predicted", base=0, marker_color='rgb(26, 118, 255)'),
                ])

bar_compare.update_layout(barmode='stack')

bar_compare.update_layout(
    autosize=False,
    height=600, 
    width=700,
    margin=dict(l=0, r=0, t=60, b=80),
    xaxis_title="Counts",
    yaxis_title="Ancestry Category",
    legend_title="Data",
)

bar1.plotly_chart(bar_compare)

ref_counts.rename(columns = {'Counts': 'Ref Panel Counts'}, inplace=True)
ref_counts.set_index('Ancestry Category', inplace=True)
new_counts.rename(columns = {'Counts': 'Predicted Counts'}, inplace=True)
new_counts.set_index('Ancestry Category', inplace=True)

total_bar_data = pd.concat([ref_counts, new_counts], axis = 1)
total_bar_data.fillna(0, inplace = True)

bar2.dataframe(total_bar_data.astype(int))
