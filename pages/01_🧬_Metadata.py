import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from functools import reduce
from PIL import Image

from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select, release_select, config_page

config_page('Metadata')

release_select()

# Pull data from different Google Cloud folders
gp2_data_bucket = get_gcloud_bucket('gp2tier2')

# Gets master key (full GP2 release or selected cohort)
master_key_path = f'{st.session_state["release_bucket"]}/clinical_data/master_key_release{st.session_state["release_choice"]}_final.csv'
master_key = blob_as_csv(gp2_data_bucket, master_key_path, sep=',')
cohort_select(master_key)

st.title(f'{st.session_state["cohort_choice"]} Metadata')

plot1, plot2 = st.columns([1,1.75])

master_key = st.session_state['master_key']  # plots full GP2 release metrics by default
master_key.rename(columns = {'age': 'Age', 'sex_for_qc': 'Sex', 'phenotype':'Phenotype'}, inplace = True)
master_key = master_key[master_key.Sex != 0]
master_key_age = master_key[master_key['Age'].notnull()]

master_key['Sex'].replace(1, 'Male', inplace = True)
master_key['Sex'].replace(2, 'Female', inplace = True)
master_key['Sex'].replace(0, 'Unknown', inplace = True)
# st.dataframe(master_key.head())

# plot1.markdown('#### Stratify Age by:')
# none = plot1.checkbox('None', )
# sex = plot1.checkbox('Sex')
# phenotype= plot1.checkbox('Phenotype')

if master_key_age.shape[0] != 0:
    plot1.markdown('#### Stratify Age by:')
    stratify = plot1.radio(
        "Stratify Age by:",
        ('None', 'Sex', 'Phenotype'), label_visibility="collapsed")

    if stratify == 'None':
        fig = px.histogram(master_key['Age'], x = 'Age', nbins = 25, color_discrete_sequence=["#332288"])
        fig.update_layout(title_text=f'<b>Age Distribution<b>')
        plot2.plotly_chart(fig)
    if stratify == 'Sex':
        fig = px.histogram(master_key, x="Age", color="Sex", nbins = 25, color_discrete_map={'Male':"#332288",'Female':"#CC6677"})
        fig.update_layout(title_text=f'<b>Age Distribution by Sex<b>')
        plot2.plotly_chart(fig)
    if stratify == 'Phenotype':
        fig = px.histogram(master_key, x="Age", color="Phenotype", nbins = 25, color_discrete_map={'Control':"#332288",'PD':"#CC6677",'Other':"#117733",'Not Reported':"#D55E00"})
        fig.update_layout(title_text=f'<b>Age Distribution by Phenotype<b>')
        plot2.plotly_chart(fig)
    
    plot1.markdown('---')

male_pheno = master_key.loc[master_key['Sex'] == 'Male', 'Phenotype']
female_pheno = master_key.loc[master_key['Sex'] == 'Female', 'Phenotype']

combined_counts = pd.DataFrame()
combined_counts['Male'] = male_pheno.value_counts()
combined_counts['Female'] = female_pheno.value_counts()
combined_counts = combined_counts.transpose()
combined_counts['Total'] = combined_counts.sum(axis=1)
combined_counts = combined_counts.fillna(0)
combined_counts = combined_counts.astype('int32')

plot1.markdown('#### Phenotype Count Split by Sex')
plot1.dataframe(combined_counts)
