import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
from hold_data import blob_as_csv, get_gcloud_bucket


# Pull data from different Google Cloud folders
frontend_bucket_name = 'frontend_app_materials'
frontend_bucket = get_gcloud_bucket(frontend_bucket_name)
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

# Save CARD and GP2 logos
card_removebg = frontend_bucket.get_blob('card-removebg.png')
card_removebg = card_removebg.download_as_bytes()
gp2_removebg = frontend_bucket.get_blob('gp2_2-removebg.png')
gp2_removebg = gp2_removebg.download_as_bytes()

st.set_page_config(
     page_title="Home",
     page_icon=card_removebg,
     layout="wide",
)

st.session_state['card_removebg'] = card_removebg
st.session_state['gp2_removebg'] = gp2_removebg 

# Background color
css = frontend_bucket.get_blob('style.css')
css = css.download_as_string()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Place logos in sidebar
sidebar1, sidebar2 = st.sidebar.columns(2)
sidebar1.image(card_removebg, use_column_width=True)
sidebar2.image(gp2_removebg, use_column_width=True)

# Main title
st.markdown("<h1 style='text-align: center; color: #0f557a; font-family: Helvetica; '>GP2 Internal Cohort Browser</h1>", unsafe_allow_html=True)

# Page formatting 
sent1, sent2, sent3 = st.columns([1,6,1])  # holds brief overview sentences
exp1, exp2, exp3 = st.columns([1, 2, 1])  # holds expander for full description

# sent2.markdown("##### Interactive tool to visualize quality control and ancestry prediction summary statistics across all GP2 cohorts. #####")
sent2.markdown("<h5 style='text-align: center; '>Interactive tool to visualize quality control and ancestry prediction summary statistics\
             across all GP2 cohorts. Please select a page marked with \U0001F9EC in the sidebar to begin.</h1>", unsafe_allow_html=True)

# # Display expander with full project description
overview = exp2.expander("Full Description", expanded=False)
with overview:
    st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

    st.markdown("## _Quality Control_")
    st.markdown('### _Sample-Level Pruning:_')
    st.markdown('Variants are pruned for missingness by case-control where P<=1e-4 to detect platform/batch differences in case-control status.\
                Next, variants are pruned for missingness by haplotype for flanking variants where P<=1e-4. Lastly, controls are filtered for HWE\
                at a threshold of 1e-4.')
    st.markdown('### _Variant-Level Pruning_')
    st.markdown('Genotypes are pruned for call rate with maximum sample genotype missingness of 0.02 (--mind 0.02). Samples which pass\
            call rate pruning are then pruned for discordant sex where samples with 0.25 <= sex F <= 0.75 are pruned. Sex F < 0.25\
            are female and Sex F > 0.75 are male. Samples that pass sex pruning are then differentiated by ancestry (refer to\
            ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness where genetic relatedness matrix (grm)\
            cutoff 0.125 is used to determine second-degree relatedness and 0.95 is used to determine duplicates. For purposes of imputation,\
            related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity where F <= -0.25 of\
            F>= 0.25.')

    st.markdown("## _Ancestry_")
    st.markdown('### _Reference Panel_')
    st.markdown('The reference panel is composed of 2975 samples from 1000 Genomes Project and an Ashkenazi Jewish reference panel\
                (Gene Expression Omnibus (GEO) database, www.ncbi.nlm.nih.gov/geo (accession no. GSE23636)) (REF) with the following\
                ancestral makeup:')
    st.markdown(
                """
                - African (AFR): 504
                - African Admixed and Caribbean (AAC): 157
                - Ashkenazi Jewish (AJ): 471
                - Central Asian (CAS): 183
                - East Asian (EAS): 504
                - European (EUR): 404
                - Finnish (FIN): 99
                - Latino/American Admixed (AMR): 347
                - Middle Eastern (MDE): 152
                - South Asian (SAS): 489
                """
                )
    st.markdown('Samples were chosen from 1000 Genomes to match the specific ancestries present in GP2. The reference panel was then\
                pruned for palindrome SNPs (A1A2= AT or TA or GC or CG). SNPs were then pruned for maf 0.05, geno 0.01, and hwe 0.0001.')

    st.markdown('### _Preprocessing_')
    st.markdown('The genotypes were pruned for geno 0.1. Common variants between the reference panel and the genotypes were extracted \
                from both the reference panel and the genotypes. Any missing genotypes were imputed using the mean of that particular\
                variant in the reference panel.')
    st.markdown('The reference panel samples were split into an 80/20 train/test set and then PCs were fit to and transformed the training\
                set using sklearn PCA.fit_transform and mean and standard deviation were extracted for scaling (𝑥/2) * (1 − (𝑥/2)) where 𝑥 is \
                the mean training set PCs. The test set was then transformed by the PCs in the training set using sklearn PCA.transform and\
                then normalized to the mean and standard deviation of the training set. Genotypes were then transformed by the same process as\
                the test set for prediction after model training.')

    st.markdown('### _UMAP + Classifier Training_')
    st.markdown('A classifier was then trained using UMAP transformations of the PCs and a linear support vector classifier using a 5-fold\
                cross-validation using an sklearn pipeline and scored for balanced accuracy with a gridsearch over the following parameters:')
    st.markdown(
                """
                - “umap__n_neighbors”: [5,20]
                - “umap__n_components”: [15,25]
                - “umap__a”: [0.75, 1.0, 1.5]
                - “umap__b”: [0.25, 0.5, 0.75]
                - “svc__C”: [0.001, 0.01, 0.1, 1, 10, 100]
                """
                )
    st.markdown('Performance varies from 95-98% balanced accuracy on the test set depending on overlapping genotypes.')

    st.markdown('### _Prediction_')
    st.markdown('Scaled PCs for genotypes are transformed using UMAP trained fitted by the training set and then predicted by the classifier. \
                Genotypes are split and output into individual ancestries. AAC and AFR labels are combined into a single category and \
                ADMIXTURE 1 (v1.3.0- https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz) is run using the –supervised \
                functionality to further divide these two categories where AFR is assigned if AFR admixture is >=90% and AAC is assigned if AFR \
                admixture is <90%')

    st.caption('**_References_**')
    st.caption('_D.H. Alexander, J. Novembre, and K. Lange. Fast model-based estimation of ancestry in unrelated individuals. Genome Research, 19:1655–1664, 2009._')

# Customize text in Expander element
hvar = """ <script>
                var elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
                elements[0].style.fontSize = 'large';
                elements[0].style.color = '#0f557a';
            </script>"""
components.html(hvar, height=0, width=0)

