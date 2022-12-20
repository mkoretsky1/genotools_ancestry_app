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
from io import StringIO

from google.cloud import bigquery
from google.cloud import storage

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

    lmap = {'r':'R','theta':'Theta'}
    smap = {'Control':'circle','PD':'circle-open'}

    fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_map=cmap, width=650, height=497, labels=lmap, symbol='phenotype', symbol_map=smap)

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

def calculate_maf(gtype_df):
    gtypes_map = {
        'AA': 0,
        'AB': 1,
        'BA': 1,
        'BB': 2,
        'NC': np.nan
        }

    gtypes = gtype_df.pivot(index='snpid', columns='iid', values='gt').replace(gtypes_map)
    
    # count only called genotypes
    N = gtypes.shape[1]-gtypes.isna().sum(axis=1)
    freq = pd.DataFrame({'freq': gtypes.sum(axis=1)/(2*N)})
    freq.loc[:,'maf'] = np.where(freq < 0.5, freq, 1-freq)
    maf_out = freq.drop(columns=['freq']).reset_index()

    return maf_out


config_page('SNP Metrics')

st.title('GP2 SNP Metrics Browser')

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)
snp_metrics_bucket_name = 'snp_metrics_db'
snp_metrics_bucket = get_gcloud_bucket(snp_metrics_bucket_name)

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

selection = f'{gene_choice}_{ancestry_choice}'
blob_name = f'hold_query/{selection}.csv'
blob = snp_metrics_bucket.blob(blob_name)

# metrics = query_metrics(metrics_query)
if selection not in st.session_state:
    if blob.exists():
        metrics = blob_as_csv(snp_metrics_bucket, blob_name, sep=',')
    else:
        metrics = query_metrics(metrics_query)
        blob = snp_metrics_bucket.blob(blob_name)
        blob.upload_from_string(metrics.to_csv(index=False), 'text/csv')
    st.session_state[selection] = metrics
else:
    metrics = st.session_state[selection]

num_sample_metrics = int(metrics.shape[0] / snps.shape[0])

with metric3:
    if ancestry_choice != 'Both':
        st.metric(f'Number of {ancestry_choice} samples with SNP metrics available:', "{:.0f}".format(num_sample_metrics))
    else:
        st.metric(f'Number of samples with SNP metrics available:', "{:.0f}".format(num_sample_metrics))

metrics_copy = metrics.copy(deep=True)
metrics_copy['snp_label'] = metrics_copy['snpid'] + ' (' + metrics_copy['chromosome'].astype(str) + ':' + metrics_copy['position'].astype(str) + ')'

if num_sample_metrics > 0:
    st.markdown('### Select SNP for Cluster Plot')
    selected_snp = st.selectbox(label='SNP', label_visibility='collapsed', options=['Select SNP!']+[snp for snp in metrics_copy['snp_label'].unique()])

    if selected_snp != 'Select SNP!':
        snp_df = metrics_copy[metrics_copy['snp_label'] == selected_snp]
        snp_df = snp_df.reset_index(drop=True)

        fig = plot_clusters(snp_df, gtype_col='gt', title=selected_snp)['fig']

        col1, col2 = st.columns([2.5,1])

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        hide_table_row_index = """<style>thead tr th:first-child {display:none} tbody th {display:none}"""
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        with col2:
            st.metric(f'GenTrain Score:', "{:.3f}".format(snp_df['gentrainscore'][0]))

            maf = calculate_maf(snp_df)
            st.metric(f'Minor Allele Frequency:', "{:.3f}".format(maf['maf'][0]))

            # st.markdown('**Control Genotype Distribution**')
            with st.expander('**Control Genotype Distribution**'):
                gt_counts = snp_df[snp_df['phenotype'] == 'Control']['gt'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
                gt_rel_counts = snp_df[snp_df['phenotype'] == 'Control']['gt'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
                gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
                st.table(gt_counts)

            # st.markdown('**PD Genotype Distribution**')
            with st.expander('**PD Genotype Distribution**'):
                gt_counts = snp_df[snp_df['phenotype'] == 'PD']['gt'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
                gt_rel_counts = snp_df[snp_df['phenotype'] == 'PD']['gt'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
                gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
                st.table(gt_counts)
    
        
