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
from hold_data import blob_as_csv, get_gcloud_bucket, gene_ancestry_select, config_page

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

    fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_map=cmap, width=650, height=497,labels={'r':'R','theta':'Theta'}, symbol='phenotype', symbol_sequence=['circle','circle-open'])

    fig.update_xaxes(range=xlim, nticks=10, zeroline=False)
    fig.update_yaxes(range=ylim, nticks=10, zeroline=False)
    
    fig.update_layout(margin=dict(r=76, t=63, b=75))

    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

    fig.update_layout(legend_title_text='Genotype')

    out_dict = {
        'fig': fig,
        'xlim': xlim,
        'ylim': ylim
    }
    
    fig.update_layout(title_text=f'<b>{title}<b>')
    
    return out_dict


config_page('SNP Metrics')

st.title('GP2 SNP Metrics Browser')

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

if gene_choice == 'GBA':
    chromosome = 1
elif gene_choice == 'SNCA':
    chromosome = 4
elif gene_choice == 'PRKN':
    chromosome = 6
else:
    chromosome = 'chromosome'

snps_query = f"select * from `{snps_table}` where chromosome={chromosome}"
snps = query_snps(snps_query)

if gene_choice == 'Both':
    metric1,metric2,metric3 = st.columns(3)
else:
    metric1,metric2,metric3 = st.columns([1.3,0.75,1])

with metric1:
    if gene_choice == 'Both':
        st.metric(f'Number of available SNPs:', "{:.0f}".format(snps.shape[0]))
    else:
        st.metric(f'Number of available SNPs associated with {gene_choice} (\U000000B1 250kb):', "{:.0f}".format(snps.shape[0]))

if ancestry_choice == 'Both':
    samples_query = f"select * from `{samples_table}`"
    samples = query_samples(samples_query)
    with metric2:
        st.metric(f'Number of samples:', "{:.0f}".format(samples.shape[0]))
    metrics_query = f"select * from `{samples_table}` join `{metrics_table}` on `{samples_table}`.iid=`{metrics_table}`.iid join `{snps_table}` on `{snps_table}`.snpid=`{metrics_table}`.snpid where `{snps_table}`.chromosome={chromosome}"
else:
    samples_query = f"select * from `{samples_table}` where `{samples_table}`.label='{st.session_state['ancestry_choice']}'"
    samples = query_samples(samples_query)
    with metric2:
        st.metric(f'Number of {ancestry_choice} samples:', "{:.0f}".format(samples.shape[0]))
    metrics_query = f"select * from `{samples_table}` join `{metrics_table}` on `{samples_table}`.iid=`{metrics_table}`.iid join `{snps_table}` on `{snps_table}`.snpid=`{metrics_table}`.snpid where `{snps_table}`.chromosome={chromosome} and `{samples_table}`.label='{ancestry_choice}'"

metrics = query_metrics(metrics_query)

num_sample_metrics = int(metrics.shape[0] / snps.shape[0])

with metric3:
    if ancestry_choice != 'Both':
        st.metric(f'Number of {ancestry_choice} samples with SNP metrics available:', "{:.0f}".format(num_sample_metrics))
    else:
        st.metric(f'Number of samples with SNP metrics available:', "{:.0f}".format(num_sample_metrics))

if num_sample_metrics > 0:
    st.markdown('### Select SNP for Cluster Plot')
    selected_snp = st.selectbox(label='SNP', label_visibility='collapsed', options=['Select SNP!']+[snp for snp in metrics['snpid'].unique()])

    col1, col2 = st.columns([2.5,1])

    if selected_snp != 'Select SNP!':
        snp_df = metrics[metrics['snpid'] == selected_snp]
        fig = plot_clusters(snp_df, gtype_col='gt', title=selected_snp)['fig']

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        hide_table_row_index = """<style>thead tr th:first-child {display:none} tbody th {display:none}"""
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        with col2:
            st.markdown('#')
            st.markdown('**Control Genotype Distribution**')
            gt_counts = snp_df[snp_df['phenotype'] == 'Control']['gt'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
            gt_rel_counts = snp_df[snp_df['phenotype'] == 'Control']['gt'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
            gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
            st.table(gt_counts)

            st.markdown('**PD Genotype Distribution**')
            gt_counts = snp_df[snp_df['phenotype'] == 'PD']['gt'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
            gt_rel_counts = snp_df[snp_df['phenotype'] == 'PD']['gt'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
            gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
            st.table(gt_counts)
