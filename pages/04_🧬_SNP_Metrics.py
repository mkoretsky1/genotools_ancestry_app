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
from hold_data import blob_as_csv, get_gcloud_bucket, chr_ancestry_select, config_page, place_logos
from io import StringIO, BytesIO

from google.cloud import bigquery
from google.cloud import storage

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
    smap = {'Control':'circle','PD':'diamond-open-dot'}

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

def snp_callback():
    st.session_state['old_snp_choice'] = st.session_state['snp_choice']
    st.session_state['snp_choice'] = st.session_state['new_snp_choice']


config_page('SNP Metrics')

st.title('GP2 SNP Metrics Browser')

sample_exp = st.expander("Description", expanded=False)
with sample_exp:
    st.markdown('To arrive at the set of variants for which SNP metrics are available, variants underwent additional pruning \
                after the steps described on the Quality Control page. Within each ancestry, variants were pruned for call rate \
                with a maximum variant genotype missingness of 0.01 (--geno 0.01), a minumum minor allele frequency of 0.01 (--maf 0.01) \
                and HWE at a thresthold of 5e-6. LD pruning was performed to find and prune any pairs of variants with r\u00b2 > 0.02 \
                in a sliding window of 1000 variants with a step size of 10 variants (--indep-pairwise 1000 10 0.02). Please note that \
                SNP Metrics are only available for the most recent GP2 release (GP2 Release 5).')

# Pull data from different Google Cloud folders
snp_metrics_bucket_name = 'redlat_gt_app_utils'
snp_metrics_bucket = get_gcloud_bucket(snp_metrics_bucket_name)

chr_ancestry_select()

chr_choice = st.session_state['chr_choice']
ancestry_choice = st.session_state['ancestry_choice']
selection = f'{ancestry_choice}_{chr_choice}'

# FIN doesn't have SNP metrics yet (> 50 samples)
if ancestry_choice ==  'FIN':
    st.error(f"SNP metrics are not yet available for FIN, because less than 50 samples have been released for this ancestry. Please \
             select a different ancestry!")

else:
    metrics_blob_name = f'gp2_snp_metrics/{ancestry_choice}/chr{chr_choice}_metrics.csv'
    maf_blob_name = f'gp2_snp_metrics/{ancestry_choice}/{ancestry_choice}_maf.afreq'
    full_maf_blob_name = f'gp2_snp_metrics/full_maf.afreq'

    if selection not in st.session_state:
        metrics = blob_as_csv(snp_metrics_bucket, metrics_blob_name, sep=',')
        st.session_state[selection] = metrics
    else:
        metrics = st.session_state[selection]

    if f'{ancestry_choice}_maf' not in st.session_state:
        maf = blob_as_csv(snp_metrics_bucket, maf_blob_name, sep='\t')
        st.session_state[f'{ancestry_choice}_maf'] = maf
    else:
        maf = st.session_state[f'{ancestry_choice}_maf']

    if 'full_maf' not in st.session_state:
        full_maf = blob_as_csv(snp_metrics_bucket, full_maf_blob_name, sep='\t')
        st.session_state[f'full_maf'] = full_maf
    else:
        full_maf = st.session_state['full_maf']


    # metrics.columns = ['snpid','r','theta','gentrainscore','gt','chromosome','position','iid','phenotype']

    metric1,metric2 = st.columns([1,1])

    num_snps = len(metrics['snpID'].unique())
    num_sample_metrics = len(metrics['Sample_ID'].unique())

    with metric1:
        st.metric(f'Number of available SNPs on Chromosome {chr_choice} for {ancestry_choice}:', "{:.0f}".format(num_snps))

    with metric2:
        st.metric(f'Number of {ancestry_choice} samples with SNP metrics available:',"{:.0f}".format(num_sample_metrics))

    metrics_copy = metrics.copy(deep=True)
    metrics_copy['snp_label'] = metrics_copy['snpID'] + ' (' + metrics_copy['chromosome'].astype(str) + ':' + metrics_copy['position'].astype(str) + ')'

    if num_sample_metrics > 0:
        snp_options = ['Select SNP!']+[snp for snp in metrics_copy['snp_label'].unique()]

        if 'snp_choice' not in st.session_state:
            st.session_state['snp_choice'] = snp_options[0]
        if 'old_snp_choice' not in st.session_state:
            st.session_state['old_snp_choice'] = ""

        if st.session_state['snp_choice'] in snp_options:
            index = snp_options.index(st.session_state['snp_choice'])
        
        if st.session_state['snp_choice'] not in snp_options:
            if ((st.session_state['snp_choice'] != 'Select SNP!') and (int(st.session_state['snp_choice'].split('(')[1].split(':')[0]) == st.session_state['old_chr_choice'])):
                index = 0
            else:
                st.error(f'SNP: {st.session_state["snp_choice"]} is not availble for {ancestry_choice}. Please choose another SNP!')
                index=0

        st.markdown('### Select SNP for Cluster Plot')

        st.session_state['snp_choice'] = st.selectbox(label='SNP', label_visibility='collapsed', options=snp_options, index=index, key='new_snp_choice', on_change=snp_callback)

        if st.session_state['snp_choice'] != 'Select SNP!':
            snp_df = metrics_copy[metrics_copy['snp_label'] == st.session_state['snp_choice']]
            snp_df = snp_df.reset_index(drop=True)

            fig = plot_clusters(snp_df, x_col='Theta', y_col='R', gtype_col='GT', title=st.session_state['snp_choice'])['fig']

            col1, col2 = st.columns([2.5,1])

            with col1:
                st.plotly_chart(fig, use_container_width=True)

                ### FOR EXPORTING CLUSTER PLOT AS PNG
                # export_button = st.button('Click here to export displayed SNP metrics cluster plot to .png!')
                # if export_button:
                #     file_name = f'cluster_plots/{ancestry_choice}_{str(snp_df["snpID"].values[0]).replace(" ","_")}.png'
                #     blob = snp_metrics_bucket.blob(file_name)

                #     buf = BytesIO()
                #     fig.write_image(buf, width=1980, height=1080)
                #     blob.upload_from_file(buf, content_type='image/png', rewind=True)

                #     st.markdown(f'Cluster plot for {st.session_state["snp_choice"]} written to {snp_metrics_bucket_name}/{file_name}')

            hide_table_row_index = """<style>thead tr th:first-child {display:none} tbody th {display:none}"""
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            with col2:
                st.metric(f'GenTrain Score:', "{:.3f}".format(snp_df['GenTrain_Score'][0]))

                within_ancestry_maf = maf[maf['ID'] == snp_df['snpID'].values[0]]
                across_ancestry_maf = full_maf[full_maf['ID'] == snp_df['snpID'].values[0]]

                st.metric(f'Minor Allele Frequency within {ancestry_choice}:', "{:.3f}".format(within_ancestry_maf['ALT_FREQS'].values[0]))
                st.metric(f'Minor Allele Frequency across ancestries:', "{:.3f}".format(across_ancestry_maf['ALT_FREQS'].values[0]))

                with st.expander('**Control Genotype Distribution**'):
                    gt_counts = snp_df[snp_df['phenotype'] == 'Control']['GT'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
                    gt_rel_counts = snp_df[snp_df['phenotype'] == 'Control']['GT'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
                    gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
                    st.table(gt_counts)

                with st.expander('**PD Genotype Distribution**'):
                    gt_counts = snp_df[snp_df['phenotype'] == 'PD']['GT'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
                    gt_rel_counts = snp_df[snp_df['phenotype'] == 'PD']['GT'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
                    gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
                    st.table(gt_counts)