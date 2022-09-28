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

def plot_3d(labeled_df, color, symbol=None, x='PC1', y='PC2', z='PC3', title=None, x_range=None, y_range=None, z_range=None):
    '''
    Parameters: 
    labeled_df (Pandas dataframe): labeled ancestry dataframe
    color (string): color of ancestry label. column name containing labels for ancestry in labeled_pcs_df
    symbol (string): symbol of secondary label (for example, predicted vs reference ancestry). default: None
    plot_out (string): filename to output filename for .png and .html plotly images
    x (string): column name of x-dimension
    y (string): column name of y-dimension
    z (string): column name of z-dimension
    title (string, optional): title of output scatterplot
    x_range (list of floats [min, max], optional): range for x-axis
    y_range (list of floats [min, max], optional): range for y-axis
    z_range (list of floats [min, max], optional): range for z-axis

    Returns:
    3-D scatterplot (plotly.express.scatter_3d). If plot_out included, will write .png static image and .html interactive to plot_out filename
        
    '''    
    fig = px.scatter_3d(
        labeled_df,
        x=x,
        y=y,
        z=z,
        color=color,
        symbol=symbol,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold,
        range_x=x_range,
        range_y=y_range,
        range_z=z_range
    )

    fig.update_traces(marker={'size': 3})

    st.plotly_chart(fig)

st.set_page_config(
    layout = 'wide'
)

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

########################  SIDE BAR #########################################
st.sidebar.markdown('**Stats/Graph Selection**', unsafe_allow_html=True)
selected_metrics = st.sidebar.selectbox(label="Stats Selection", options=['SNPs', 'Samples', 'Both'])
selected_metrics_1 = st.sidebar.selectbox(label="PCA selection", options=['Reference PCA', 'Projected PCA', 'Both'])

geno_path = f'data/GP2_QC_round3_MDGAP-QSBB'
ref_labels = f'data/ref_panel_ancestry.txt'
out_path = f'data/GP2_QC_round3_MDGAP-QSBB'

print(os.getcwd())
outdir = os.path.dirname(out_path)
plot_dir = f'{outdir}/plot_ancestry'

st.markdown(f'<p class="small-font">Cohort: {out_path.split("/")[-1].split("_")[-1]}</p>', unsafe_allow_html=True)

ref_common_snps = pd.read_csv(f'data/ref_common_snps.common_snps', sep='\s+', header=None)
geno_common_snps = pd.read_csv(f'{out_path}_common_snps.common_snps', sep='\s+', header=None)
geno_fam = pd.read_csv(f'{geno_path}.fam', sep='\s+', header=None)
ref_fam = pd.read_csv(f'data/ref_common_snps.fam', sep='\s+', header=None)

if (selected_metrics == 'SNPs') | (selected_metrics == 'Both'):
    st.markdown(f'<p class="small-font">Number of SNPs used to train prediction model: {ref_common_snps.shape[0]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="small-font">Number of overlapping SNPs: {geno_common_snps.shape[0]}</p>', unsafe_allow_html=True)

if (selected_metrics == 'Samples') | (selected_metrics == 'Both'):
    st.markdown(f'<p class="small-font">Number of samples for prediction: {geno_fam.shape[0]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="small-font">Train set size: {round(ref_fam.shape[0]*0.8)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="small-font">Test set size: {round(ref_fam.shape[0]*0.2)}</p>', unsafe_allow_html=True)

ref_pca_path = f'{out_path}_labeled_ref_pca.txt'
ref_pca = pd.read_csv(ref_pca_path, sep='\s+')
new_pca_path = f'{out_path}_projected_new_pca.txt'
new_pca = pd.read_csv(new_pca_path, sep='\s+')
total_pca = pd.concat([ref_pca, new_pca], axis=0)

if (selected_metrics_1 == 'Reference PCA') | (selected_metrics_1 == 'Both'):
    plot_3d(ref_pca, 'label')

if (selected_metrics_1 == 'Projected PCA') | (selected_metrics_1 == 'Both'):
    plot_3d(total_pca, 'label')

