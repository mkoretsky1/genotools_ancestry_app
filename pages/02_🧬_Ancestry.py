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
from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select

st.set_page_config(page_title = "Ancestry", layout = 'wide')

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

master_key = blob_as_csv(gp2_sample_bucket, f'master_key_release3_final.csv', sep=',')
cohort_select(master_key)

if ('cohort_choice' in st.session_state) and ('upload_data_path' not in st.session_state):
    st.markdown(f'### Cohort: {st.session_state["cohort_choice"]}')
    master_key = st.session_state['master_key']
# else:
#     geno_path = st.session_state['upload_data_path']
#     ref_labels = f'ref_panel_ancestry.txt'
#     out_path = st.session_state['upload_data_path']
#     st.markdown(f'### **Cohort: {out_path}**')

tabPCA, tabPredStats, tabPie, tabAdmix = st.tabs(["Ancestry Prediction", "Model Performance", "Ancestry Distribution", "Admixture Populations"])

# if ('sample_data_path' not in st.session_state) and ('upload_data_path' not in st.session_state):
#     st.error('Error: Please use the Upload Data page to either submit .bed/.bim/.fam files or choose a sample cohort!')

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
                hover_name="IID",
                color_discrete_map={'AFR': '#1f77b4',
                                    'SAS': '#ff7f0e',
                                    'EAS': '#2ca02c',
                                    'EUR':'#d62728',
                                    'AMR':'#9467bd',
                                    'AJ':'#8c564b',
                                    'AAC':'#e377c2',
                                    'CAS':'#8ceca8',
                                    'MDE':'#bcbd22',
                                    'FIN':'#17becf'}
            )

    fig.update_traces(marker={'size': 3})
    st.plotly_chart(fig)

def plot_pie(df):
    pie_chart = px.pie(df, names = 'Ancestry Category', values = 'Proportion', hover_name = 'Ancestry Category',  color="Ancestry Category", 
                    color_discrete_map={'AFR': '#1f77b4',
                                        'SAS': '#ff7f0e',
                                        'EAS': '#2ca02c',
                                        'EUR':'#d62728',
                                        'AMR':'#9467bd',
                                        'AJ':'#8c564b',
                                        'AAC':'#e377c2',
                                        'CAS':'#8ceca8',
                                        'MDE':'#bcbd22',
                                        'FIN':'#17becf'})
    pie_chart.update_layout(showlegend = True, width=500,height=500)
    st.plotly_chart(pie_chart)

if 'cohort_choice' not in st.session_state:
    st.error('Error: Please use the drop-down menu on the sidebar to choose a sample cohort!')
else:
    with tabPCA:
        # st.markdown(f'## **PCA Analysis**')

        ref_pca = blob_as_csv(gp2_sample_bucket, f'reference_pcs.csv', sep=',')
        proj_pca = blob_as_csv(gp2_sample_bucket, f'projected_pcs.csv', sep=',')

        proj_pca = proj_pca.drop(columns=['label'], axis=1)

        proj_pca_cohort = proj_pca.merge(master_key[['GP2sampleID','label']], how='inner', left_on=['IID'], right_on=['GP2sampleID'])
        proj_pca_cohort = proj_pca_cohort.drop(columns=['GP2sampleID'], axis=1)
        proj_pca_cohort['plot_label'] = 'Predicted'

        ref_pca['plot_label'] = ref_pca['label']

        total_pca = pd.concat([ref_pca, proj_pca_cohort], axis=0)
        new_labels = proj_pca_cohort['label']

        pca_col1, pca_col2 = st.columns([1.5,3])
        st.markdown('---')
        col1, col2 = st.columns([1.5, 3])

        combined = proj_pca_cohort[['IID', 'label']]
        combined_labelled = combined.rename(columns={'label': 'Predicted Ancestry'})
        holdValues = combined['label'].value_counts().rename_axis('Predicted Ancestry').reset_index(name='Counts')

        with pca_col1:
            # st.markdown('### Reference Panel vs. Projected PCA')
            st.markdown(f'### Reference Panel vs. {st.session_state["cohort_choice"]} PCA')
            with st.expander("Description"):
                st.write('Select an Ancestry Category below to display only the Predicted samples within that label.')

            gb = GridOptionsBuilder.from_dataframe(holdValues)
            # gb.configure_pagination(paginationAutoPageSize=True)
            # gb.configure_side_bar()
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
            gridOptions = gb.build()

            grid_response = AgGrid(
                        holdValues,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=True,
                        theme='streamlit',
                        enable_enterprise_modules=True, 
                        width = '100%' ,
                        height = 350
                    )
            selected = grid_response['selected_rows'] 
            selected_df = pd.DataFrame(selected) # selected rows from AgGrid passed to new df

        with pca_col2:
            if not selected_df.empty:
                selected_pca = proj_pca_cohort.copy()
                selectionList = []

                for selections in selected_df['Predicted Ancestry']:
                    selectionList.append(selections)
                
                selected_pca.drop(selected_pca[np.logical_not(selected_pca['label'].isin(selectionList))].index, inplace = True)
                
                for items in selectionList:
                    selected_pca.replace({items: 'Predicted'}, inplace = True)

                total_pca_selected = pd.concat([ref_pca, selected_pca], axis=0)
                plot_3d(total_pca_selected, 'label')
            else:
                plot_3d(total_pca, 'plot_label')

        with col1:
            # st.markdown('### Projected PCA')
            st.markdown(f'### {st.session_state["cohort_choice"]} PCA')
            with st.expander("Description"):
                st.write('All Predicted samples and their respective labels are listed below.')
            gb = GridOptionsBuilder.from_dataframe(combined_labelled)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gridOptions = gb.build()

            grid_response = AgGrid(
                        combined_labelled,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=True,
                        theme='streamlit',
                        enable_enterprise_modules=True, 
                        width='100%',
                        height = 350
                    )
        with col2: 
            plot_3d(proj_pca_cohort, 'label')


    with tabPredStats:
        st.markdown(f'## **Model Accuracy**')
        confusion_matrix = blob_as_csv(gp2_sample_bucket, 'confusion_matrix.csv', sep=',')
        confusion_matrix.set_index(confusion_matrix.columns, inplace = True)
        
        cm_matrix = confusion_matrix.to_numpy()
        total = cm_matrix.sum()
        accuracy=(np.trace(cm_matrix))/total

        heatmap1, heatmap2 = st.columns([2, 1])
        fig = px.imshow(confusion_matrix, labels=dict(x="Predicted Ancestry", y="Reference Panel Ancestry", color="Count"), text_auto=True)
        heatmap1.plotly_chart(fig)

        with heatmap2:
            st.markdown('### Test Set Performance')
            st.metric('Classification Accuracy:', "{:.3f}".format(round(accuracy, 3)))
            # st.metric("Test Sensitivity:", )
            # st.metric("Test Specificity:", )

    # with tabPie:
    #     st.markdown('### **Reference Panel Ancestry**')
    #     pie1, pie2 = st.columns([2, 1])

    #     ref_pca = blob_as_csv(gp2_sample_bucket, f'reference_pcs.csv', sep=',')

    #     df_ancestry_counts = ref_pca['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
    #     ref_counts = ref_pca['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
    #     ref_combo = pd.merge(df_ancestry_counts, ref_counts, on='Ancestry Category')

    #     pie2.dataframe(ref_combo)

    #     with pie1:
    #         plot_pie(df_ancestry_counts)

    #     st.markdown('---')

    #     st.markdown('### **Predicted Ancestry**')

    #     pie3, pie4 = st.columns([2, 1])

    #     df_new_counts = master_key['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
    #     new_counts = master_key['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
    #     new_combo = pd.merge(df_new_counts, new_counts, on='Ancestry Category')

    #     pie4.dataframe(new_combo)

    #     with pie3:
    #         plot_pie(df_new_counts)

    with tabPie:
        pie1, pie2, pie3 = st.columns([2,1,2])
        p1, p2, p3 = st.columns([2,2,2])

        ref_pca = blob_as_csv(gp2_sample_bucket, f'reference_pcs.csv', sep=',')

        df_ancestry_counts = ref_pca['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
        ref_counts = ref_pca['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
        ref_combo = pd.merge(df_ancestry_counts, ref_counts, on='Ancestry Category')
        ref_combo.rename(columns = {'Proportion': 'Ref Panel Proportion', 'Counts': 'Ref Panel Counts'}, inplace = True)

        df_new_counts = master_key['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
        new_counts = master_key['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
        new_combo = pd.merge(df_new_counts, new_counts, on='Ancestry Category')
        new_combo.rename(columns = {'Proportion': 'Predicted Proportion', 'Counts': 'Predicted Counts'}, inplace = True)

        pie_table = pd.merge(ref_combo, new_combo, on='Ancestry Category')

        with pie1:
            st.markdown('### **Reference Panel Ancestry**')
            plot_pie(df_ancestry_counts)
            # st.dataframe(ref_combo)

        with pie3:
            # st.markdown('### **Predicted Ancestry**')
            st.markdown(f'### {st.session_state["cohort_choice"]} Predicted Ancestry')
            plot_pie(df_new_counts)
            # st.dataframe(new_combo)

        with p2:
            st.dataframe(pie_table[['Ancestry Category', 'Ref Panel Counts', 'Predicted Counts']])

    with tabAdmix:
        frontend_bucket_name = 'frontend_app_materials'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)

        st.markdown('## **Reference Panel Admixture Populations**')

        ref_admix = blob_as_csv(frontend_bucket, 'ref_panel_admixture.txt')

        admix_pop_info = ref_admix['ancestry']
        admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10']]

        admix_plot = frontend_bucket.get_blob('refpanel_admix.png')
        admix_plot = admix_plot.download_as_bytes()
        st.image(admix_plot)

        # admix_chart1, admix_chart2, admix_chart3 = st.columns([0.5,2,0.5])
        # admix_chart2.dataframe(ref_admix)
        # st.dataframe(ref_admix)

        gb = GridOptionsBuilder.from_dataframe(ref_admix)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        grid_response = AgGrid(
                        ref_admix,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=True,
                        theme='streamlit',
                        enable_enterprise_modules=True, 
                        width='100%',
                        height = 400
                    )
