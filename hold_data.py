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
from utils.dependencies import check_plink, check_plink2, check_admixture

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'genotools-02f64a1e10be.json'

plink_exec = check_plink()
plink2_exec = check_plink2()
admix_exec = check_admixture()

def blob_to_csv(bucket, path, header='infer'):
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep='\s+', header=header)
    return df

storage_client = storage.Client(project='genotools')
bucket_name = 'frontend_app_data'
bucket = storage_client.get_bucket(bucket_name)
st.session_state['bucket'] = bucket

out_path = f'GP2_QC_round3_MDGAP-QSBB'
new_pca_path = f'{out_path}_projected_new_pca.txt'
# new_pca = pd.read_csv(new_pca_path, sep='\s+')
# new_labels = pd.read_csv(f'{out_path}_umap_linearsvc_predicted_labels.txt', delimiter = "\t")
new_pca = blob_to_csv(bucket, new_pca_path)
new_labels = blob_to_csv(bucket, f'{out_path}_umap_linearsvc_predicted_labels.txt')

combined = pd.merge(new_pca, new_labels, on='IID')
combined.rename(columns = {'IID': 'Sample ID', 'label_y': 'Predicted Ancestry'}, inplace = True)

if 'combined' not in st.session_state:
    st.session_state['combined'] = combined

