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

def get_gcloud_bucket(bucket_name):  # gets folders from google cloud
    storage_client = storage.Client(project='genotools')
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

# Sidebar selector (on every page)
def cohort_select(master_key):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)

    if 'cohort_choice' in st.session_state:  # changes selector label to chosen cohort
        selected_metrics = st.sidebar.selectbox(label = 'Cohort Selection', label_visibility = 'collapsed', options=[st.session_state['cohort_choice'], 'GP2 Release 3 FULL']+
                            [study for study in master_key['study'].unique()])
    else:  # defaults to GP2 Release 3 FULL cohort
        selected_metrics = st.sidebar.selectbox(label = 'Cohort Selection', label_visibility = 'collapsed', options=['GP2 Release 3 FULL']+[study for study in master_key['study'].unique()])
    
    st.session_state['cohort_choice'] = selected_metrics  # can change selected cohort at any time

    if selected_metrics == 'GP2 Release 3 FULL':
        st.session_state['master_key'] = master_key
    else:
        master_key_cohort = master_key[master_key['study'] == selected_metrics]
        st.session_state['master_key'] = master_key_cohort  # subsets master key to only include selected cohort

    # checking for pruned samples
    if 1 in st.session_state.master_key['pruned'].value_counts():
        pruned_samples = st.session_state.master_key['pruned'].value_counts()[1]
    else:
        pruned_samples = 0
    
    total_count = st.session_state['master_key'].shape[0]

    st.sidebar.metric("", selected_metrics)
    st.sidebar.metric("Number of Samples in Dataset:", f'{total_count:,}')
    st.sidebar.metric("Number of Samples After Pruning:", f'{(total_count-pruned_samples):,}')
