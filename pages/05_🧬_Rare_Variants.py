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
from hold_data import blob_as_csv, get_gcloud_bucket, config_page, place_logos, rv_select
from io import StringIO

config_page('Rare Variants')

st.title('GP2 Rare Variant Browser')

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)
snp_metrics_bucket_name = 'snp_metrics_db'
snp_metrics_bucket = get_gcloud_bucket(snp_metrics_bucket_name)

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

# Gets rare variant data
rv_data = blob_as_csv(gp2_sample_bucket, f'gp2_RV_browser_input.csv', sep=',')
rv_select(rv_data)

st.markdown(len(st.session_state['rv_cohort_choice']))

