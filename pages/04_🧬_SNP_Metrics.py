import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from hold_data import blob_as_csv, get_gcloud_bucket, gene_ancestry_select

from google.cloud import bigquery

@st.cache
def query_snps(snps_query):
    snps = client.query(snps_query).to_dataframe()
    return snps

@st.cache
def query_samples(samples_query):
    samples = client.query(samples_query).to_dataframe()
    return samples

@st.cache
def query_metrics(metrics_query):
    metrics = client.query(metrics_query).to_dataframe()
    return metrics

def plot_clusters(df, x_col='theta', y_col='r', gtype_col='gt', title='snp plot'):
    d3 = px.colors.qualitative.D3

    cmap = {
        'AA': d3[0],
        'AB': d3[1],
        'BA': d3[1],
        'BB': d3[2],
        'NC': d3[3]
    }

    # gtypes_list = (df[gtype_col].unique())
    xmin, xmax = df[x_col].min(), df[x_col].max()
    ymin, ymax = df[y_col].min(), df[y_col].max()

    xlim = [xmin-.1, xmax+.1]
    ylim = [ymin-.1, ymax+.1]

    fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_map=cmap, width=650, height=497)

    fig.update_xaxes(range=xlim, nticks=10)
    fig.update_yaxes(range=ylim, nticks=10)
    
    fig.update_layout(margin=dict(r=76, t=63, b=75),)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

    fig.update_layout(legend_title_text='')

    out_dict = {
        'fig': fig,
        'xlim': xlim,
        'ylim': ylim
    }
    
    fig.update_layout(title_text=title)
    
    return out_dict

if 'gp2_removebg' in st.session_state:
    st.set_page_config(
        page_title="Ancestry",
        page_icon=st.session_state.gp2_bg,
        layout="wide",
    )
else: 
    st.set_page_config(
        page_title="Ancestry",
        layout="wide"
    )

st.title('SNP Metrics Browser')

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

gene_ancestry_select()

gene_choice = st.session_state['gene_choice']
ancestry_choice = st.session_state['ancestry_choice']

client = bigquery.Client()

db_name = 'genotools.snp_metrics'
snps_table = f'{db_name}.snps'
samples_table = f'{db_name}.samples'
metrics_table = f'{db_name}.metrics'

if gene_choice == 'SNCA':
    chromosome = 4
elif gene_choice == 'PRKN':
    chromosome = 6
else:
    chromosome = 'chromosome'

snps_query = f"select * from `{snps_table}` where chromosome={chromosome}"
snps = query_snps(snps_query)

if gene_choice == 'Both':
        st.markdown(f'Number of available SNPs: {snps.shape[0]}')
else:
    st.markdown(f'Number of available SNPs associated with {gene_choice} (\U000000B1 250kb): {snps.shape[0]}')

if ancestry_choice == 'Both':
    samples_query = f"select * from `{samples_table}`"
    samples = query_samples(samples_query)
    st.markdown(f'Number of samples: {samples.shape[0]}')
    metrics_query = f"select * from `{samples_table}` join `{metrics_table}` on `{samples_table}`.iid=`{metrics_table}`.iid join `{snps_table}` on `{snps_table}`.snpid=`{metrics_table}`.snpid where `{snps_table}`.chromosome={chromosome}"
else:
    samples_query = f"select * from `{samples_table}` where `{samples_table}`.label='{st.session_state['ancestry_choice']}'"
    samples = query_samples(samples_query)
    st.markdown(f'Number of {ancestry_choice} samples: {samples.shape[0]}')
    metrics_query = f"select * from `{samples_table}` join `{metrics_table}` on `{samples_table}`.iid=`{metrics_table}`.iid join `{snps_table}` on `{snps_table}`.snpid=`{metrics_table}`.snpid where `{snps_table}`.chromosome={chromosome} and `{samples_table}`.label='{ancestry_choice}'"

metrics = query_metrics(metrics_query)

num_sample_metrics = int(metrics.shape[0] / snps.shape[0])

if ancestry_choice != 'Both':
        st.markdown(f'Number of {ancestry_choice} samples with SNP metrics available: {num_sample_metrics}')
else:
    st.markdown(f'Number of samples with SNP metrics available: {num_sample_metrics}')

if num_sample_metrics > 0:
    st.markdown('SNP Selection for Cluster Plot')
    selected_snp = st.selectbox(label='SNP', label_visibility='collapsed', options=['Select SNP!']+[snp for snp in metrics['snpid'].unique()])
    
    col1, col2, col3 = st.columns([0.2,1,0.2])

    if selected_snp != 'Select SNP!':
        plot_df = metrics[metrics['snpid'] == selected_snp]
        fig = plot_clusters(plot_df, gtype_col='gt', title=selected_snp)['fig']
        with col2:
            st.plotly_chart(fig)