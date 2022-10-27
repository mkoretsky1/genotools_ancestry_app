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
from streamlit.components.v1 import html

from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos

from hold_data import blob_as_csv, get_gcloud_bucket

st.set_page_config(page_title = "Upload Data", layout = 'wide')

bucket_name = 'frontend_app_data'
bucket = get_gcloud_bucket(bucket_name)

head_1, head_2, title, head_3 = st.columns([0.3, 0.3, 1, 0.3])

gp2 = bucket.get_blob('gp2_2.jpg')
gp2 = gp2.download_as_bytes()
head_1.image(gp2, width=120)

card = bucket.get_blob('card.jpeg')
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

    pkl = bucket.get_blob('GP2_QC_round2_callrate_sex_ancestry_umap_linearsvc_ancestry_model.pkl')
    st.markdown('<p class="small-font">MODEL TRAINED</p>', unsafe_allow_html=True)  
    st.markdown(f'<p class="small-font">{str(pkl.updated).split(".")[0]}</p>', unsafe_allow_html=True)

#########################  SIDE BAR #########################################

st.sidebar.markdown('**Upload your own data!**', unsafe_allow_html=True)
uploaded_data = st.sidebar.file_uploader('Accepts PLINK file formats', accept_multiple_files=True, type=['bed', 'bim', 'fam'])
st.session_state['uploaded_data'] = uploaded_data  # can use across all pages

if len(uploaded_data) > 3:
    st.error('Error! Please limit submissions to 3 files (1 of each .bed/.bim/.fam)')
elif uploaded_data:
    needExtensions = ['bed', 'bim', 'fam']
    extensions = []
    allThree = False

    for files in uploaded_data:
        holdExtension = files.name.split('.')
        extensions.append(holdExtension[1])
        if holdExtension[1] == 'bed':
            bed_file = files
        elif holdExtension[1] == 'bim':
            bim_file = files
        elif holdExtension[1] == 'fam':
            fam_file = files

        if set(extensions) == set(needExtensions):
            allThree = True
            st.success('All .bed/.bim/.fam files are successfully uploaded!')

    if not allThree:
        st.error('Error! Please submit at least 1 of each PLINK file type: .bed, .bim, .fam')

    if allThree:
        # st.write(uploaded_data)
        file_summary, sample_summary, model_choice = st.columns([2, 1, 2])
        file_summary.markdown(f'### **File Paths**')

        file_prefixes = []
        for files in uploaded_data:
            file_summary.text(files.name)
            holdName= files.name.split('.')
            geno_path = holdName[0]
            if geno_path not in file_prefixes:
                file_prefixes.append(geno_path)
        
        with file_summary:
            with st.expander("More File Info:"):
                st.write(uploaded_data)

        file_summary.markdown(f'### **File Prefix**')
        file_summary.text(*file_prefixes)

        # fam_df = pd.read_csv(fam_file)
        # bim_df = pd.read_csv(bim_file)

        fam_df = blob_as_csv(st.session_state.bucket, fam_file)
        bim_df = blob_as_csv(st.session_state.bucket, bim_file)

        # sample_summary.text(f"Number of Samples in Dataset: {len(fam_df.index)}")
        # sample_summary.text(f"Number of SNPs in Dataset: {len(bim_df.index)}")

        sample_summary.metric("Number of Samples in Dataset:", len(fam_df.index))
        sample_summary.metric("Number of SNPs in Dataset:", len(bim_df.index))

        # if sample_summary.button('Confirm Dataset'):
        #     with model_choice:
        #         form = st.form("my_form", clear_on_submit = False)
        #         form.markdown(f'### **Model Creation**')
        #         form.markdown(f'###### Please select:')
        #         training = form.radio("Would you like to",
        #             ('predict using a pretrained model', 'train a new model'), horizontal= True, label_visibility = 'collapsed')
        #         submit = form.form_submit_button('Submit')
        #         if submit:
        #              st.write(training)

        with model_choice:
            form = st.form("my_form")
            form.markdown(f'### **Model Creation**')
            form.markdown(f'###### Please select:')
            training = form.radio("Would you like to",
                    ('predict using a pretrained model', 'train a new model'), horizontal= True, label_visibility = 'collapsed')
            submit = form.form_submit_button('Submit')
            if submit:
                form.markdown(f'#### *Thank you for your submission!*')

        # if sample_summary.button('Confirm Dataset'):
            # model_choice.markdown(f'### **Model Creation**')
            # model_choice.markdown(f'###### Please select:')
            # training = model_choice.radio("Would you like to",
            #     ('predict using a pretrained model', 'train a new model'), horizontal= True, label_visibility = 'collapsed')


        # plink_cmd1 = f'{plink_exec} --bfile data/{geno_path}\
        #             --missing\
        #             --freq\
        #             --out data/{geno_path}-temp-summary'

        # shell_do(plink_cmd1)

        # with open(f'data/{geno_path}-temp-summary.log') as summaryLog:
        #     lines = summaryLog.readlines()
        #     st.write(lines)
