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
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import datetime
from hold_data import blob_as_csv, get_gcloud_bucket


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
        range_z=z_range,
        hover_name="IID"
    )

    fig.update_traces(marker={'size': 3})

    st.plotly_chart(fig)

def pca_details(labeled_df, color, ref_pca = None, pred_df = None, text_df = None, predShow = True):
    col1, col2, col3 = st.columns([2, 2, 2])

    if predShow:
        full_df = st.session_state.combined
        combined = st.session_state.combined[['Sample ID', 'Predicted Ancestry']]
        holdValues = combined['Predicted Ancestry'].value_counts().rename_axis('Predicted Ancestry Labels').reset_index(name='Counts')

        with col1:
            gb = GridOptionsBuilder.from_dataframe(combined)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gridOptions = gb.build()

            grid_response = AgGrid(
                        combined,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=False,
                        theme='streamlit',
                        enable_enterprise_modules=True, 
                        width='100%',
                        height = 300
                    )

        with col2:
            gb = GridOptionsBuilder.from_dataframe(holdValues)
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
            gridOptions = gb.build()

            grid_response = AgGrid(
                        holdValues,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=False,
                        theme='streamlit',
                        enable_enterprise_modules=True, 
                        width = '100%' ,
                        height = 300
                    )
            
            data = grid_response['data']
            selected = grid_response['selected_rows'] 
            selected_df = pd.DataFrame(selected) # selected rows from AgGrid passed to new df

        if not selected_df.empty:
            selected_pca = full_df.copy()
            selectionList = []

            for selections in selected_df['Predicted Ancestry Labels']:
                selectionList.append(selections)
            
            selected_pca.drop(selected_pca[np.logical_not(selected_pca['Predicted Ancestry'].isin(selectionList))].index, inplace = True)
            selected_pca.rename(columns = {'Predicted Ancestry': 'label'}, inplace = True)
            for items in selectionList:
                selected_pca.replace({items: 'Predicted'}, inplace = True)

            total_pca_selected = pd.concat([ref_pca, selected_pca], axis=0)
            plot_3d(total_pca_selected, 'label')

st.set_page_config(page_title = "PCA Analysis", layout = 'wide')

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

st.markdown(f'## **PCA Analysis**')

if ('sample_data_path' not in st.session_state) and ('upload_data_path' not in st.session_state):
    st.error('Error! Please use the Upload Data page to either submit .bed/.bim/.fam files or choose a sample cohort!')

else:
    if ('sample_data_path' in st.session_state) and ('upload_data_path' not in st.session_state):
        geno_path = st.session_state['sample_data_path']
        ref_labels = f'ref_panel_ancestry.txt'
        out_path = st.session_state['sample_data_path']
        st.markdown(f'### **Cohort: {out_path.split("/")[-1].split("_")[-1]}**')
    else:
        geno_path = st.session_state['upload_data_path']
        ref_labels = f'ref_panel_ancestry.txt'
        out_path = st.session_state['upload_data_path']
        st.markdown(f'### **Cohort: {out_path}**')

    ref_common_snps = blob_as_csv(ref_panel_bucket, 'ref_common_snps.common_snps', header=None)
    ref_fam = blob_as_csv(ref_panel_bucket, 'ref_common_snps.fam', header=None)
    geno_common_snps = blob_as_csv(gp2_sample_bucket, f'{out_path}_common_snps.common_snps', header=None)
    geno_fam = blob_as_csv(gp2_sample_bucket, f'{geno_path}.fam', header=None)

    tab1, tab2, tab3 = st.tabs(["SNPs", "Samples", "Both"])

    with tab1:
        metric_cols1, metric_cols2, metric_cols3 = st.columns(3)
        metric_cols1.metric('SNPs for Model Training', ref_common_snps.shape[0])
        metric_cols2.metric('Overlapping SNPs', geno_common_snps.shape[0])
    with tab2:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric('Samples for Prediction', geno_fam.shape[0])
        metric_col2.metric('Train Set Size', round(ref_fam.shape[0]*0.8))
        metric_col3.metric('Test Set Size', round(ref_fam.shape[0]*0.2))
    with tab3:
        metric_cols1, metric_cols2, metric_cols3 = st.columns(3)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_cols1.metric('SNPs for Model Training', ref_common_snps.shape[0])
        metric_cols2.metric('Overlapping SNPs', geno_common_snps.shape[0])
        metric_col1.metric('Samples for Prediction', geno_fam.shape[0])
        metric_col2.metric('Train Set Size', round(ref_fam.shape[0]*0.8))
        metric_col3.metric('Test Set Size', round(ref_fam.shape[0]*0.2))


    selected_metrics_1 = st.selectbox(label = 'PCA Selection', label_visibility = 'collapsed', options=['Click to select PCA Plot...', 'Reference PCA', 'Projected PCA', 'Both'])

    ref_pca = blob_as_csv(gp2_sample_bucket, f'{out_path}_labeled_ref_pca.txt')
    new_pca = blob_as_csv(gp2_sample_bucket, f'{out_path}_projected_new_pca.txt')

    total_pca = pd.concat([ref_pca, new_pca], axis=0)
    total_pca_copy = total_pca.replace({'new' : 'Predicted'})
    new_labels = blob_as_csv(gp2_sample_bucket, f'{out_path}_umap_linearsvc_predicted_labels.txt')

    combined = pd.merge(new_pca, new_labels, on='IID')
    combined.rename(columns = {'IID': 'Sample ID', 'label_y': 'Predicted Ancestry'}, inplace = True)
    st.session_state['combined'] = combined

    if (selected_metrics_1 == 'Reference PCA') | (selected_metrics_1 == 'Both'):
        plot_3d(ref_pca, 'label')
        pca_details(ref_pca, 'label', predShow = False)

    if (selected_metrics_1 == 'Projected PCA') | (selected_metrics_1 == 'Both'):
        plot_3d(total_pca_copy, 'label')
        pca_details(total_pca_copy, 'label', ref_pca, new_pca, new_labels)
