import os
import sys
import subprocess
# from git import blob
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

def blob_as_csv(bucket, path, sep='\s+', header='infer'):
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep=sep, header=header)
    return df

def get_gcloud_bucket(bucket_name):
    storage_client = storage.Client(project='genotools')
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

def cohort_select(master_key):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)
    selected_metrics = st.sidebar.selectbox(label = 'Cohort Selection', label_visibility = 'collapsed', options=['Click to select Cohort...','GP2 Release 3 FULL']+[study for study in master_key['study'].unique()])

    if selected_metrics != 'Click to select Cohort...':
        st.session_state['cohort_choice'] = selected_metrics

        if selected_metrics == 'GP2 Release 3 FULL':
            st.session_state['master_key'] = master_key

        else:
            master_key_cohort = master_key[master_key['study'] == selected_metrics]
            st.session_state['master_key'] = master_key_cohort

        pruned_samples = st.session_state.master_key['pruned'].value_counts()[1]
        total_count = st.session_state['master_key'].shape[0]

        st.sidebar.metric("", selected_metrics)
        st.sidebar.metric("Number of Samples in Dataset:", total_count)
        st.sidebar.metric("Number of Samples After Pruning:", (total_count-pruned_samples))
