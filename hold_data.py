import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from PIL import Image
import datetime
from io import StringIO

from google.cloud import storage

from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'genotools-02f64a1e10be.json'

# Functions auto loaded and used in every page
def blob_as_csv(bucket, path, sep='\s+', header='infer'):  # reads in file from google cloud folder
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep=sep, header=header)
    return df

def get_gcloud_bucket(bucket_name):  # gets folders from Google Cloud
    storage_client = storage.Client(project='genotools')
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

def config_page(title):
    if 'gp2_bg' in st.session_state:
        st.set_page_config(
            page_title=title,
            page_icon=st.session_state.gp2_bg,
            layout="wide",
        )
    else: 
        frontend_bucket_name = 'frontend_app_materials'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)
        gp2_bg = frontend_bucket.get_blob('gp2_2.jpg')
        gp2_bg = gp2_bg.download_as_bytes()
        st.session_state['gp2_bg'] = gp2_bg
        st.set_page_config(
            page_title=title,
            page_icon=gp2_bg,
            layout="wide"
        )


def place_logos():
    sidebar1, sidebar2 = st.sidebar.columns(2)
    if ('card_removebg' in st.session_state) and ('redlat' in st.session_state):
        sidebar1.image(st.session_state.card_removebg, use_column_width=True)
        sidebar2.image(st.session_state.gp2_removebg, use_column_width=True)
        st.sidebar.image(st.session_state.redlat, use_column_width=True)
    else:
        frontend_bucket_name = 'frontend_app_materials'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)
        card_removebg = frontend_bucket.get_blob('card-removebg.png')
        card_removebg = card_removebg.download_as_bytes()
        gp2_removebg = frontend_bucket.get_blob('gp2_2-removebg.png')
        gp2_removebg = gp2_removebg.download_as_bytes()
        redlat = frontend_bucket.get_blob('Redlat.png')
        redlat = redlat.download_as_bytes()
        st.session_state['card_removebg'] = card_removebg
        st.session_state['gp2_removebg'] = gp2_removebg
        st.session_state['redlat'] = redlat
        sidebar1.image(card_removebg, use_column_width=True)
        sidebar2.image(gp2_removebg, use_column_width=True)
        st.sidebar.image(redlat, use_column_width=True)

# Sidebar selector (on every page)
def release_callback():
    st.session_state['old_release_choice'] = st.session_state['release_choice']
    st.session_state['release_choice'] = st.session_state['new_release_choice']

def release_select():
    st.sidebar.markdown('### **Choose a release!**')
    options = ['GP2 Release 3']

    if 'release_choice' not in st.session_state:
        st.session_state['release_choice'] = options[0]
    if 'old_release_choice' not in st.session_state:
        st.session_state['old_release_choice'] = ""
    
    st.session_state['release_choice'] = st.sidebar.selectbox(label='Release Selection', label_visibility='collapsed', options=options, index=options.index(st.session_state['release_choice']), key='new_release_choice', on_change=release_callback)

def cohort_callback():
    st.session_state['old_cohort_choice'] = st.session_state['cohort_choice']
    st.session_state['cohort_choice'] = st.session_state['new_cohort_choice']

def cohort_select(master_key):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)

    options=['GP2 Release 3 FULL']+[study for study in master_key['study'].unique()]

    if 'cohort_choice' not in st.session_state:
        st.session_state['cohort_choice'] = options[0]
    if 'old_cohort_choice' not in st.session_state:
        st.session_state['old_cohort_choice'] = ""

    st.session_state['cohort_choice'] = st.sidebar.selectbox(label = 'Cohort Selection', label_visibility = 'collapsed', options=options, index=options.index(st.session_state['cohort_choice']), key='new_cohort_choice', on_change=cohort_callback)

    if st.session_state['cohort_choice'] == 'GP2 Release 3 FULL':
        st.session_state['master_key'] = master_key
    else:
        master_key_cohort = master_key[master_key['study'] == st.session_state['cohort_choice']]
        st.session_state['master_key'] = master_key_cohort  # subsets master key to only include selected cohort

    # Check for pruned samples
    if 1 in st.session_state.master_key['pruned'].value_counts():
        pruned_samples = st.session_state.master_key['pruned'].value_counts()[1]
    else:
        pruned_samples = 0
    
    total_count = st.session_state['master_key'].shape[0]

    st.sidebar.metric("", st.session_state['cohort_choice'])
    st.sidebar.metric("Number of Samples in Dataset:", f'{total_count:,}')
    st.sidebar.metric("Number of Samples After Pruning:", f'{(total_count-pruned_samples):,}')

    # Place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()

def gene_callback():
    st.session_state['old_gene_choice'] = st.session_state['gene_choice']
    st.session_state['gene_choice'] = st.session_state['new_gene_choice']

def ancestry_callback():
    st.session_state['old_ancestry_choice'] = st.session_state['ancestry_choice']
    st.session_state['ancestry_choice'] = st.session_state['new_ancestry_choice']

def gene_ancestry_select():
    st.sidebar.markdown('### **Choose a gene!**', unsafe_allow_html=True)

    gene_options=['GBA','SNCA', 'PRKN','Both']

    if 'gene_choice' not in st.session_state:
        st.session_state['gene_choice'] = gene_options[0]
    if 'old_gene_choice' not in st.session_state:
        st.session_state['old_gene_choice'] = ""

    st.session_state['gene_choice'] = st.sidebar.selectbox(label = 'Gene Selection', label_visibility = 'collapsed', options=gene_options, index=gene_options.index(st.session_state['gene_choice']), key='new_gene_choice', on_change=gene_callback)

    st.sidebar.markdown('### **Choose an Ancestry!**', unsafe_allow_html=True)

    ancestry_options=['AAC','AFR','Both']

    if 'ancestry_choice' not in st.session_state:
        st.session_state['ancestry_choice'] = ancestry_options[0]
    if 'old_ancestry_choice' not in st.session_state:
        st.session_state['old_ancestry_choice'] = ""

    st.session_state['ancestry_choice'] = st.sidebar.selectbox(label = 'Ancestry Selection', label_visibility = 'collapsed', options=ancestry_options, index=ancestry_options.index(st.session_state['ancestry_choice']), key='new_ancestry_choice', on_change=ancestry_callback)

    # Place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()

def rv_cohort_callback():
    st.session_state['old_rv_cohort_choice'] = st.session_state['rv_cohort_choice']
    st.session_state['rv_cohort_choice'] = st.session_state['new_rv_cohort_choice']

def method_callback():
    st.session_state['old_method_choice'] = st.session_state['method_choice']
    st.session_state['method_choice'] = st.session_state['new_method_choice']

def rv_gene_callback():
    st.session_state['old_rv_gene_choice'] = st.session_state['rv_gene_choice']
    st.session_state['rv_gene_choice'] = st.session_state['new_rv_gene_choice']

def rv_select(rv_data):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)

    rv_cohort_options = [i for i in rv_data['Study code'].unique()]

    if 'rv_cohort_choice' not in st.session_state:
        st.session_state['rv_cohort_choice'] = None
    if 'old_rv_cohort_choice' not in st.session_state:
        st.session_state['old_rv_cohort_choice'] = ""

    st.session_state['rv_cohort_choice'] = st.sidebar.multiselect(label = 'Cohort Selection', label_visibility = 'collapsed', options=rv_cohort_options, default=st.session_state['rv_cohort_choice'], key='new_rv_cohort_choice', on_change=rv_cohort_callback)


    st.sidebar.markdown('### **Choose a discovery method!**', unsafe_allow_html=True)

    method_options = [i for i in rv_data['Methods'].unique()]

    if 'method_choice' not in st.session_state:
        st.session_state['method_choice'] = None
    if 'old_method_choice' not in st.session_state:
        st.session_state['old_method_choice'] = ""

    st.session_state['method_choice'] = st.sidebar.multiselect(label = 'Method Selection', label_visibility = 'collapsed', options=method_options, default=st.session_state['method_choice'], key='new_method_choice', on_change=method_callback)


    st.sidebar.markdown('### **Choose a gene!**', unsafe_allow_html=True)

    rv_gene_options = [i for i in rv_data['Gene'].unique()]

    if 'rv_gene_choice' not in st.session_state:
        st.session_state['rv_gene_choice'] = None
    if 'old_rv_gene_choice' not in st.session_state:
        st.session_state['old_rv_gene_choice'] = ""

    st.session_state['rv_gene_choice'] = st.sidebar.multiselect(label = 'Gene Selection', label_visibility = 'collapsed', options=rv_gene_options, default=st.session_state['rv_gene_choice'], key='new_rv_gene_choice', on_change=rv_gene_callback)

    # Place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()