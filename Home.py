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
import hold_data
from io import StringIO

from google.cloud import storage

from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos

from hold_data import get_gcloud_bucket

st.set_page_config(page_title = "Home", layout = 'wide')

frontend_bucket_name = 'frontend_app_materials'
frontend_bucket = get_gcloud_bucket(frontend_bucket_name)

ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

# Background color
css = frontend_bucket.get_blob('style.css')
css = css.download_as_string()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

####################### HEAD ##############################################

head_1, head_2, title, head_3 = st.columns([0.3, 0.3, 1, 0.3])

gp2 = frontend_bucket.get_blob('gp2_2.jpg')
gp2 = gp2.download_as_bytes()
head_1.image(gp2, width=120)

card = frontend_bucket.get_blob('card.jpeg')
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

    pkl = ref_panel_bucket.get_blob('GP2_QC_round2_callrate_sex_ancestry_umap_linearsvc_ancestry_model.pkl')
    st.markdown('<p class="small-font">MODEL TRAINED</p>', unsafe_allow_html=True)  
    st.markdown(f'<p class="small-font">{str(pkl.updated).split(".")[0]}</p>', unsafe_allow_html=True)
