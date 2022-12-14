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
from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select


if 'gp2_removebg' in st.session_state:
    st.set_page_config(
        page_title="Ancestry",
        page_icon=st.session_state.gp2_bg,
        layout="wide",
    )
else: 
    st.set_page_config(
        page_title="Ancestry",
        layout="wide"
    )

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
ref_panel_bucket_name = 'ref_panel'
ref_panel_bucket = get_gcloud_bucket(ref_panel_bucket_name)

# Gets master key (full GP2 release or selected cohort)
master_key = blob_as_csv(gp2_sample_bucket, f'master_key_release3_final.csv', sep=',')
cohort_select(master_key)

# Update when re-add Upload Data feature
if ('cohort_choice' in st.session_state) and ('upload_data_path' not in st.session_state):
    st.title(f'Cohort: {st.session_state["cohort_choice"]}')
    master_key = st.session_state['master_key']
# else:
#     geno_path = st.session_state['upload_data_path']
#     ref_labels = f'ref_panel_ancestry.txt'
#     out_path = st.session_state['upload_data_path']
#     st.markdown(f'### **Cohort: {out_path}**')

# if ('sample_data_path' not in st.session_state) and ('upload_data_path' not in st.session_state):
#     st.error('Error: Please use the Upload Data page to either submit .bed/.bim/.fam files or choose a sample cohort!')


# Tab navigator for different parts of Ancestry Method
tabPCA, tabPredStats, tabPie, tabAdmix, tabMethods = st.tabs(["Ancestry Prediction", "Model Performance", "Ancestry Distribution",\
                                                    "Admixture Populations", "Method Description"])

# Plots 3D PCA
def plot_3d(labeled_df, color, symbol=None, x='PC1', y='PC2', z='PC3', title=None, x_range=None, y_range=None, z_range=None):
    '''
    Parameters: 
    labeled_df (Pandas dataframe): labeled ancestry dataframe
    color (string): color of ancestry label. column name containing labels for ancestry in labeled_pcs_df
    symbol (string): symbol of secondary label (for example, predicted vs reference ancestry). default: None
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
                color_discrete_map={'AFR': "#88CCEE",
                                    'SAS': "#CC6677",
                                    'EAS': "#DDCC77",
                                    'EUR':"#117733",
                                    'AMR':"#332288",
                                    'AJ': "#D55E00",
                                    'AAC':"#999933",
                                    'CAS':"#882255",
                                    'MDE':"#661100",
                                    'FIN': "#F0E442"}
            )

    fig.update_traces(marker={'size': 3})
    st.plotly_chart(fig)

# Plots pie chart
def plot_pie(df):
    '''
    Parameters:
    df (Pandas dataframe): dataframe with ancestry categories and respective counts
    
    Returns:
    interactive pie chart (plotly.express.pie)
    '''
    pie_chart = px.pie(df, names = 'Ancestry Category', values = 'Proportion', hover_name = 'Ancestry Category',  color="Ancestry Category", 
                    color_discrete_map={'AFR': "#88CCEE",
                                        'SAS': "#CC6677",
                                        'EAS': "#DDCC77",
                                        'EUR':"#117733",
                                        'AMR':"#332288",
                                        'AJ': "#D55E00",
                                        'AAC':"#999933",
                                        'CAS':"#882255",
                                        'MDE':"#661100",
                                        'FIN':"#F0E442"})
    pie_chart.update_layout(showlegend = True, width=500,height=500)
    st.plotly_chart(pie_chart)

if 'cohort_choice' not in st.session_state:
    st.error('Error: Please use the drop-down menu on the sidebar to choose a sample cohort!')
else:
    with tabPCA:
        ref_pca = blob_as_csv(gp2_sample_bucket, f'reference_pcs.csv', sep=',')
        proj_pca = blob_as_csv(gp2_sample_bucket, f'projected_pcs.csv', sep=',')

        proj_pca = proj_pca.drop(columns=['label'], axis=1)

        # Projected PCAs will have label "Predicted", Reference panel is labeled by ancestry 
        proj_pca_cohort = proj_pca.merge(master_key[['GP2sampleID','label']], how='inner', left_on=['IID'], right_on=['GP2sampleID'])
        proj_pca_cohort = proj_pca_cohort.drop(columns=['GP2sampleID'], axis=1)
        proj_pca_cohort['plot_label'] = 'Predicted'
        ref_pca['plot_label'] = ref_pca['label']

        total_pca = pd.concat([ref_pca, proj_pca_cohort], axis=0)
        new_labels = proj_pca_cohort['label']

        pca_col1, pca_col2 = st.columns([1.5,3])
        st.markdown('---')
        col1, col2 = st.columns([1.5, 3])

        # Get actual ancestry labels of each sample in Projected PCAs instead of "Predicted" for all samples
        combined = proj_pca_cohort[['IID', 'label']]
        combined_labelled = combined.rename(columns={'label': 'Predicted Ancestry'})
        holdValues = combined['label'].value_counts().rename_axis('Predicted Ancestry').reset_index(name='Counts')

        with pca_col1:
            st.markdown(f'### Reference Panel vs. {st.session_state["cohort_choice"]} PCA')
            with st.expander("Description"):
                st.write('Select an Ancestry Category below to display only the Predicted samples within that label.')

            gb = GridOptionsBuilder.from_dataframe(holdValues)
            # gb.configure_pagination(paginationAutoPageSize=True)
            # gb.configure_side_bar()
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
            gridOptions = gb.build()

            # Selectable dataframe to plot specific ancestry categories of predicted samples 
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
            selected_df = pd.DataFrame(selected)  # selected rows from AgGrid passed to new df

        with pca_col2:
            # If category selected, plots Projected PCA samples in that category under label "Predicted"
            if not selected_df.empty:
                selected_pca = proj_pca_cohort.copy()
                selectionList = []

                for selections in selected_df['Predicted Ancestry']:
                    selectionList.append(selections)
                
                selected_pca.drop(selected_pca[np.logical_not(selected_pca['label'].isin(selectionList))].index, inplace = True)
                
                for items in selectionList:  # subsets Projected PCA by selected categories
                    selected_pca.replace({items: 'Predicted'}, inplace = True)

                total_pca_selected = pd.concat([ref_pca, selected_pca], axis=0)
                plot_3d(total_pca_selected, 'label')
            else:
                plot_3d(total_pca, 'plot_label')  # if no category selected, plots all samples of Projected PCA with "Predicted" label

        with col1:
            st.markdown(f'### {st.session_state["cohort_choice"]} PCA')
            with st.expander("Description"):
                st.write('All Predicted samples and their respective labels are listed below.')
            gb = GridOptionsBuilder.from_dataframe(combined_labelled)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gridOptions = gb.build()

            # Non-selectable dataframe: lists all Predicted subjects and their respective ancestry predictions
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
            plot_3d(proj_pca_cohort, 'label')  # only plots PCA of predicted samples


    with tabPredStats:
        st.markdown(f'## **Model Accuracy**')
        confusion_matrix = blob_as_csv(gp2_sample_bucket, 'confusion_matrix.csv', sep=',')
        confusion_matrix.set_index(confusion_matrix.columns, inplace = True)

        tp = np.diag(confusion_matrix)
        col_sum = confusion_matrix.sum(axis=0)
        row_sum = confusion_matrix.sum(axis=1)

        class_recall = np.array(tp/row_sum)
        class_precision = np.array(tp/col_sum)

        balanced_accuracy = np.mean(class_recall)
        margin_of_error = 1.96 * np.sqrt((balanced_accuracy*(1-balanced_accuracy))/sum(col_sum))

        precision = np.mean(class_precision)

        f1 = np.mean(2 * ((class_recall * class_precision)/(class_recall + class_precision)))

        heatmap1, heatmap2 = st.columns([2, 1])

        with heatmap1:
            st.markdown('### Confusion Matrix')
            fig = px.imshow(confusion_matrix, labels=dict(x="Predicted Ancestry", y="Reference Panel Ancestry", color="Count"), text_auto=True)
            st.plotly_chart(fig)

        # Plots heatmap of confusion matrix from Testing
        with heatmap2:
            st.markdown('### Test Set Performance')
            st.markdown('#')
            st.metric('Balanced Accuracy:', "{:.3f} \U000000B1 {:.3f}".format(round(balanced_accuracy, 3), round(margin_of_error, 3)))
            st.markdown('#')
            st.metric('Precision:', "{:.3f}".format(round(precision, 3)))
            st.markdown('#')
            st.metric('F1 Score:', "{:.3f}".format(round(f1, 3)))

    with tabPie:
        # Plots ancestry breakdowns of Predicted Samples vs. Reference Panel samples
        pie1, pie2, pie3 = st.columns([2,1,2])
        p1, p2, p3 = st.columns([2,2,2])

        ref_pca = blob_as_csv(gp2_sample_bucket, f'reference_pcs.csv', sep=',')

        # Get dataframe of counts per ancestry category for reference panel
        df_ancestry_counts = ref_pca['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
        ref_counts = ref_pca['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
        ref_combo = pd.merge(df_ancestry_counts, ref_counts, on='Ancestry Category')
        ref_combo.rename(columns = {'Proportion': 'Ref Panel Proportion', 'Counts': 'Ref Panel Counts'}, inplace = True)

        # Gets dataframe of counts per ancestry category for predicted samples
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
            st.markdown(f'### {st.session_state["cohort_choice"]} Predicted Ancestry')
            plot_pie(df_new_counts)
            # st.dataframe(new_combo)

        # Displays dataframe of Reference Panel Counts vs. Predicted Counts per Ancestry category
        with p2:
            st.dataframe(pie_table[['Ancestry Category', 'Ref Panel Counts', 'Predicted Counts']])

    with tabAdmix:
        frontend_bucket_name = 'frontend_app_materials'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)

        st.markdown('## **Reference Panel Admixture Populations**')

        ref_admix = blob_as_csv(frontend_bucket, 'ref_panel_admixture.txt')

        admix_pop_info = ref_admix['ancestry']
        admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10']]

        # Admix plot saved as PNG in admix_graph.py to avoid wait time for plot generation
        admix_plot = frontend_bucket.get_blob('refpanel_admix.png')
        admix_plot = admix_plot.download_as_bytes()
        st.image(admix_plot)

        gb = GridOptionsBuilder.from_dataframe(ref_admix)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        # Non-selectable dataframe: allows advanced filtering of subjects' admixture results
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
    with tabMethods:
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
