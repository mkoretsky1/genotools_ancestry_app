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
import hold_data
from io import StringIO
from google.cloud import storage
from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos
from utils.dependencies import check_plink, check_plink2, check_admixture

st.set_page_config(page_title = "Home", layout = 'wide')

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'genotools-02f64a1e10be.json'

# plink_exec = check_plink()
# plink2_exec = check_plink2()
# admix_exec = check_admixture()

# def blob_to_csv(bucket, path, header='infer'):
#     blob = bucket.get_blob(path)
#     blob = blob.download_as_bytes()
#     blob = str(blob, 'utf-8')
#     blob = StringIO(blob)
#     df = pd.read_csv(blob, sep='\s+', header=header)
#     return df

# storage_client = storage.Client(project='genotools')
# bucket_name = 'frontend_app_data'
# bucket = storage_client.get_bucket(bucket_name)
# st.session_state['bucket'] = bucket

# out_path = f'GP2_QC_round3_MDGAP-QSBB'
# new_pca_path = f'{out_path}_projected_new_pca.txt'
# # new_pca = pd.read_csv(new_pca_path, sep='\s+')
# # new_labels = pd.read_csv(f'{out_path}_umap_linearsvc_predicted_labels.txt', delimiter = "\t")
# new_pca = blob_to_csv(bucket, new_pca_path)
# new_labels = blob_to_csv(bucket, f'{out_path}_umap_linearsvc_predicted_labels.txt')

# combined = pd.merge(new_pca, new_labels, on='IID')
# combined.rename(columns = {'IID': 'Sample ID', 'label_y': 'Predicted Ancestry'}, inplace = True)

# if 'combined' not in st.session_state:
#     st.session_state['combined'] = combined

# Background color
css = st.session_state.bucket.get_blob('style.css')
css = css.download_as_string()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

####################### HEAD ##############################################

head_1, head_2, title, head_3 = st.columns([0.3, 0.3, 1, 0.3])

gp2 = st.session_state.bucket.get_blob('gp2_2.jpg')
gp2 = gp2.download_as_bytes()
head_1.image(gp2, width=120)

card = st.session_state.bucket.get_blob('card.jpeg')
card = card.download_as_bytes()
head_2.image(card, width=120)

with title:
    st.markdown("""
    <style>
    .big-font {
        font-family:Helvetica; color:#0f557a; font-size:34px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">GenoTools Ancestry Prediction</p>', unsafe_allow_html=True)
    
with head_3:
    def modification_date(filename):
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)
    st.markdown("""
    <style>
    .small-font {
        font-family:Helvetica; color:#0f557a; font-size:16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    pkl = st.session_state.bucket.get_blob('GP2_QC_round2_callrate_sex_ancestry_umap_linearsvc_ancestry_model.pkl')
    st.markdown('<p class="small-font">MODEL TRAINED</p>', unsafe_allow_html=True)  
    st.markdown(f'<p class="small-font">{str(pkl.updated).split(".")[0]}</p>', unsafe_allow_html=True)
