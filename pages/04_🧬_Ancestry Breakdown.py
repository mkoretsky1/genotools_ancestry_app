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
from hold_data import blob_as_csv, get_gcloud_bucket

st.set_page_config(page_title = "Ancestry Breakdown", layout = 'wide')

st.markdown('## **Reference Panel Ancestry**')

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

# if ('sample_data_path' not in st.session_state) and ('upload_data_path' not in st.session_state):
#     st.error('Error! Please use the Upload Data page to either submit .bed/.bim/.fam files or choose a sample cohort!')

if 'sample_data_path' not in st.session_state:
    st.error('Error! Please use the Upload Data page to choose a sample cohort!')

else:
    if ('sample_data_path' in st.session_state) and ('upload_data_path' not in st.session_state):
        out_path = st.session_state['sample_data_path']
    else:
        out_path = st.session_state['upload_data_path']

    new_pca = blob_as_csv(gp2_sample_bucket, f'{out_path}_projected_new_pca.txt')
    new_labels = blob_as_csv(gp2_sample_bucket, f'{out_path}_umap_linearsvc_predicted_labels.txt')

    combined = pd.merge(new_pca, new_labels, on='IID')
    combined.rename(columns = {'IID': 'Sample ID', 'label_y': 'Predicted Ancestry'}, inplace = True)

    pie1, pie2 = st.columns([2, 1])

    out_path = f'GP2_QC_round3_MDGAP-QSBB'
    ref_pca_path = f'{out_path}_labeled_ref_pca.txt'
    ref_pca = blob_as_csv(gp2_sample_bucket, ref_pca_path)

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

    pie3, pie4 = st.columns([2, 1])

    df_new_counts = combined['Predicted Ancestry'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
    new_counts = combined['Predicted Ancestry'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
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
