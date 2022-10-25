#upload some libraries
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
import sys
 
# adding GenoTools/QC to the system path
sys.path.insert(0, '/home/kuznetsovn2/Desktop/GenoTools/QC')
 
# importing QC utils
from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos
from utils.dependencies import check_plink, check_plink2, check_admixture

plink_exec = check_plink()
plink2_exec = check_plink2()
admix_exec = check_admixture()

st.set_page_config(page_title = "Home", layout = 'wide')

out_path = f'data/GP2_QC_round3_MDGAP-QSBB'
new_pca_path = f'{out_path}_projected_new_pca.txt'
new_pca = pd.read_csv(new_pca_path, sep='\s+')
new_labels = pd.read_csv(f'{out_path}_umap_linearsvc_predicted_labels.txt', delimiter = "\t")

combined = pd.merge(new_pca, new_labels, on='IID')
combined.rename(columns = {'IID': 'Sample ID', 'label_y': 'Predicted Ancestry'}, inplace = True)

if 'combined' not in st.session_state:
    st.session_state['combined'] = combined


##Background color
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css(f'data/style.css')

####################### HEAD ##############################################

head_1, head_2, title, head_3 = st.columns([0.3, 0.3, 1, 0.3])

gp2 = Image.open(f'data/gp2_2.jpg')
head_1.image(gp2, width=120)

card = Image.open(f'data/card.jpeg')
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

    date = modification_date(f'data/GP2_QC_round2_callrate_sex_ancestry_umap_linearsvc_ancestry_model.pkl')
    st.markdown('<p class="small-font">MODEL TRAINED</p>', unsafe_allow_html=True)  
    st.markdown(f'<p class="small-font">{date}</p>', unsafe_allow_html=True)
