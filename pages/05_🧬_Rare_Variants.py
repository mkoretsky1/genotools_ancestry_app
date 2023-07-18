import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from hold_data import blob_as_csv, get_gcloud_bucket, config_page, rv_select

config_page('GP2 Rare Variant Browser')

# release_select()

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gt_app_utils'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

# Gets rare variant data
rv_data = blob_as_csv(gp2_sample_bucket, f'gp2_RV_browser_input.csv', sep=',')
rv_select(rv_data)

st.title('GP2 Rare Variant Browser')

# Filter df based on the user selection 
if len(st.session_state['rv_cohort_choice'])>0:      
    rv_data_selected = rv_data[rv_data['Study code'].isin(st.session_state['rv_cohort_choice'])].reset_index(drop=True)
else:
    rv_data_selected = rv_data

if len(st.session_state['method_choice'])>0: 
    rv_data_selected = rv_data_selected[rv_data_selected['Methods'].isin(st.session_state['method_choice'])].reset_index(drop=True)
else:
    rv_data_selected = rv_data_selected

if len(st.session_state['rv_gene_choice'])>0: 
    rv_data_selected = rv_data_selected[rv_data_selected['Gene'].isin(st.session_state['rv_gene_choice'])].reset_index(drop=True)
else:
    rv_data_selected = rv_data_selected

st.dataframe(rv_data_selected, hide_index=True, use_container_width=True)

