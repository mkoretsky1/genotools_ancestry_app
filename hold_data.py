import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

from google.cloud import storage

# functions used on every pages

# reads in file from google cloud folder
def blob_as_csv(bucket, path, sep='\s+', header='infer'):
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep=sep, header=header)
    return df

# gets folders from Google Cloud
def get_gcloud_bucket(bucket_name): 
    storage_client = storage.Client(project='gp2-release-terra')
    bucket = storage_client.bucket(bucket_name, user_project='gp2-release-terra')
    return bucket

# config page with gp2 logo in browser tab
def config_page(title):
    if 'gp2_bg' in st.session_state:
        st.set_page_config(
            page_title=title,
            page_icon=st.session_state.gp2_bg,
            layout="wide",
        )
    else: 
        frontend_bucket_name = 'gt_app_utils'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)
        gp2_bg = frontend_bucket.get_blob('gp2_2.jpg')
        gp2_bg = gp2_bg.download_as_bytes()
        st.session_state['gp2_bg'] = gp2_bg
        st.set_page_config(
            page_title=title,
            page_icon=gp2_bg,
            layout="wide"
        )

# load and place sidebar logos
def place_logos():
    sidebar1, sidebar2 = st.sidebar.columns(2)
    if ('card_removebg' in st.session_state) and ('redlat' in st.session_state):
        sidebar1.image(st.session_state.card_removebg, use_column_width=True)
        sidebar2.image(st.session_state.gp2_removebg, use_column_width=True)
        st.sidebar.image(st.session_state.redlat, use_column_width=True)
    else:
        frontend_bucket_name = 'gt_app_utils'
        frontend_bucket = get_gcloud_bucket(frontend_bucket_name)
        card_removebg = frontend_bucket.get_blob('card-removebg.png')
        card_removebg = card_removebg.download_as_bytes()
        gp2_removebg = frontend_bucket.get_blob('gp2_2-removebg.png')
        gp2_removebg = gp2_removebg.download_as_bytes()
        redlat = frontend_bucket.get_blob('Redlat.png')
        redlat = redlat.download_as_bytes()
        st.session_state['card_removebg'] = card_removebg
        st.session_state['gp2_removebg'] = gp2_removebg
        st.session_state['redlat'] = redlat
        sidebar1.image(card_removebg, use_column_width=True)
        sidebar2.image(gp2_removebg, use_column_width=True)
        st.sidebar.image(redlat, use_column_width=True)

# sidebar selectors
        
# release callback for session state
def release_callback():
    st.session_state['old_release_choice'] = st.session_state['release_choice']
    st.session_state['release_choice'] = st.session_state['new_release_choice']

# release select
def release_select():
    st.sidebar.markdown('### **Choose a release!**')
    options = [7, 6, 5, 4, 3, 2, 1]

    # set selector default
    if 'release_choice' not in st.session_state:
        st.session_state['release_choice'] = options[0]
    if 'old_release_choice' not in st.session_state:
        st.session_state['old_release_choice'] = ""
    
    # dynamic change of release with selector
    st.session_state['release_choice'] = st.sidebar.selectbox(label='Release Selection', label_visibility='collapsed', options=options, index=options.index(st.session_state['release_choice']), key='new_release_choice', on_change=release_callback)

    # folder name based on release selection
    release_folder_dict = {1:'release1_29112021', 2:'release2_06052022', 3:'release3_31102022', 4:'release4_14022023', 
                           5:'release5_11052023', 6:'release6_21122023', 7:'release7_30042024'}
    st.session_state['release_bucket'] = release_folder_dict[st.session_state['release_choice']]

# cohort callback for session state
def cohort_callback():
    st.session_state['old_cohort_choice'] = st.session_state['cohort_choice']
    st.session_state['cohort_choice'] = st.session_state['new_cohort_choice']

# cohort select
def cohort_select(master_key):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)

    # get cohort options
    options=[f'GP2 Release {st.session_state["release_choice"]} FULL']+[study for study in master_key['study'].unique()]
    full_release_options=[f'GP2 Release {i} FULL' for i in range(1,8)] 

    # set selector default
    if 'cohort_choice' not in st.session_state:
        st.session_state['cohort_choice'] = options[0]

    # error message for when cohort is not available in a previous release
    if st.session_state['cohort_choice'] not in options:
        # exclude full releases
        if (st.session_state['cohort_choice'] not in full_release_options):
            st.error(f"Cohort: {st.session_state['cohort_choice']} not available for GP2 Release {st.session_state['release_choice']}. \
                    Displaying GP2 Release {st.session_state['release_choice']} FULL instead!")
        st.session_state['cohort_choice'] = options[0]

    if 'old_cohort_choice' not in st.session_state:
        st.session_state['old_cohort_choice'] = ""

    # dynamic change of cohort with selector
    st.session_state['cohort_choice'] = st.sidebar.selectbox(label = 'Cohort Selection', label_visibility = 'collapsed', options=options, index=options.index(st.session_state['cohort_choice']), key='new_cohort_choice', on_change=cohort_callback)

    if st.session_state['cohort_choice'] == f'GP2 Release {st.session_state["release_choice"]} FULL':
        st.session_state['master_key'] = master_key
    else:
        master_key_cohort = master_key[master_key['study'] == st.session_state['cohort_choice']]
        # subsets master key to only include selected cohort
        st.session_state['master_key'] = master_key_cohort 

    # check for pruned samples
    if 1 in st.session_state.master_key['pruned'].value_counts():
        pruned_samples = st.session_state.master_key['pruned'].value_counts()[1]
    else:
        pruned_samples = 0
    
    total_count = st.session_state['master_key'].shape[0]

    # sidebar metrics
    st.sidebar.metric("", st.session_state['cohort_choice'])
    st.sidebar.metric("Number of Samples in Dataset:", f'{total_count:,}')
    st.sidebar.metric("Number of Samples After Pruning:", f'{(total_count-pruned_samples):,}')

    # place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()

# metadata ancestry callback for session state
def meta_ancestry_callback():
    st.session_state['old_meta_ancestry_choice'] = st.session_state['meta_ancestry_choice']
    st.session_state['meta_ancestry_choice'] = st.session_state['new_meta_ancestry_choice']

# metadata ancestry select
def meta_ancestry_select():
    st.markdown('### **Choose an ancestry!**')
    master_key = st.session_state['master_key']

    # set options
    meta_ancestry_options = ['All'] + [label for label in master_key['label'].dropna().unique()]

    # set default
    if 'meta_ancestry_choice' not in st.session_state:
        st.session_state['meta_ancestry_choice'] = meta_ancestry_options[0]

    # error message when ancestry not available in selected cohort
    if st.session_state['meta_ancestry_choice'] not in meta_ancestry_options:
        st.error(f"No samples with {st.session_state['meta_ancestry_choice']} ancestry in {st.session_state['cohort_choice']}. \
                 Displaying all ancestries instead!")
        st.session_state['meta_ancestry_choice'] = meta_ancestry_options[0]

    if 'old_meta_ancestry_choice' not in st.session_state:
        st.session_state['old_chr_choice'] = ""
    
    # dynamic change of ancestry with selector
    st.session_state['meta_ancestry_choice'] = st.selectbox(label='Ancestry Selection', label_visibility = 'collapsed', options=meta_ancestry_options, index=meta_ancestry_options.index(st.session_state['meta_ancestry_choice']), key='new_meta_ancestry_choice', on_change=meta_ancestry_callback)

# admixture ancestry callback for session state
def admix_ancestry_callback():
    st.session_state['old_admix_ancestry_choice'] = st.session_state['admix_ancestry_choice']
    st.session_state['admix_ancestry_choice'] = st.session_state['new_admix_ancestry_choice']

# admixture ancestry select
def admix_ancestry_select():
    st.markdown('### **Choose an ancestry!**')
    master_key = st.session_state['master_key']

    # set options
    admix_ancestry_options = ['All'] + [label for label in master_key['label'].dropna().unique()]

    # set default
    if 'admix_ancestry_choice' not in st.session_state:
        st.session_state['admix_ancestry_choice'] = admix_ancestry_options[0]
    if 'old_admix_ancestry_choice' not in st.session_state:
        st.session_state['old_chr_choice'] = ""
    
    # dynamic change of ancestry with selector
    st.session_state['admix_ancestry_choice'] = st.selectbox(label='Ancestry Selection', label_visibility = 'collapsed', options=admix_ancestry_options, index=admix_ancestry_options.index(st.session_state['admix_ancestry_choice']), key='new_admix_ancestry_choice', on_change=admix_ancestry_callback)

# snp metrics chromosome callback for session state
def chr_callback():
    st.session_state['old_chr_choice'] = st.session_state['chr_choice']
    st.session_state['chr_choice'] = st.session_state['new_chr_choice']

# snp metrics ancestry callback for session state
def ancestry_callback():
    st.session_state['old_ancestry_choice'] = st.session_state['ancestry_choice']
    st.session_state['ancestry_choice'] = st.session_state['new_ancestry_choice']

# snp metrics chromosome/ancestry selector
def chr_ancestry_select():
    st.sidebar.markdown('### **Select SNPs By:**', unsafe_allow_html=True)

    # set chromosome options
    chr_options=[i for i in range(1,23)]
    ndd_genes = pd.read_csv('data/NDD_gene_regions.csv') # will need to change out for Google Cloud

    # select_options = ['Chromosome', 'NDD-Related Gene Interval', 'Chromosome + NDD Gene']
    # st.session_state['snp_select_choice'] = st.sidebar.selectbox(label = 'SNP Selection Method', label_visibility = 'collapsed', options=select_options, key='snp_selection_method')
    chr_choice = st.sidebar.checkbox('Chromosome', value = True)
    gene_choice = st.sidebar.checkbox('NDD-Related Gene Interval')

    if chr_choice:
        if 'chr_choice' not in st.session_state:
            st.session_state['chr_choice'] = chr_options[0]
        if 'old_chr_choice' not in st.session_state:
            st.session_state['old_chr_choice'] = ""

        if 'ndd_gene_choice' not in st.session_state:
            st.session_state['ndd_gene_choice'] = ""

        st.sidebar.markdown('### **Choose a chromosome!**', unsafe_allow_html=True)
        st.session_state['chr_choice'] = st.sidebar.selectbox(label = 'Chromosome Selection', label_visibility = 'collapsed', options=chr_options, index=chr_options.index(st.session_state['chr_choice']), key='new_chr_choice', on_change=chr_callback)
    if gene_choice:
        if chr_choice:
            gene_options = ndd_genes.NAME[ndd_genes.CHR == st.session_state['chr_choice']].unique()
            st.sidebar.markdown(f"### **Choose an NDD-related gene in chromosome {st.session_state['chr_choice']}!**", unsafe_allow_html=True)
        else:
            gene_options = ndd_genes.NAME.unique()
            st.sidebar.markdown('### **Choose an NDD-related gene!**', unsafe_allow_html=True)

        if 'ndd_gene_choice' not in st.session_state:
            st.session_state['ndd_gene_choice'] = gene_options[0]
        if 'old_ndd_gene_choice' not in st.session_state:
            st.session_state['old_ndd_gene_choice'] = ""

        st.session_state['ndd_gene_choice'] = st.sidebar.selectbox(label = 'NDD-Related Gene Selection', label_visibility = 'collapsed', options=gene_options, key='new_ndd_gene_choice', on_change=ndd_gene_callback)
        # review where to put the CHR and positions for the chosen gene
        position_df = ndd_genes[ndd_genes.NAME == st.session_state['ndd_gene_choice']]
        st.sidebar.markdown('_**Gene Location:**_')
        for row in range(len(position_df)):
            st.sidebar.markdown(f'_CHR {position_df.CHR.values[row]}:{position_df.START.values[row]}-{position_df.STOP.values[row]}_')

    # dynamic change of chromosome with selector
    # st.session_state['chr_choice'] = st.sidebar.selectbox(label = 'Chromosome Selection', label_visibility = 'collapsed', options=chr_options, index=chr_options.index(st.session_state['chr_choice']), key='new_chr_choice', on_change=chr_callback)

    st.sidebar.markdown('### **Choose an Ancestry!**', unsafe_allow_html=True)

    # set ancestry options
    ancestry_options=['AAC','AFR','AJ','AMR','CAH','CAS','EAS','EUR','FIN','MDE','SAS']

    # set default ancestry
    if 'ancestry_choice' not in st.session_state:
        st.session_state['ancestry_choice'] = ancestry_options[0]
    if 'old_ancestry_choice' not in st.session_state:
        st.session_state['old_ancestry_choice'] = ""

    # dynamic change of ancestry with selector
    st.session_state['ancestry_choice'] = st.sidebar.selectbox(label = 'Ancestry Selection', label_visibility = 'collapsed', options=ancestry_options, index=ancestry_options.index(st.session_state['ancestry_choice']), key='new_ancestry_choice', on_change=ancestry_callback)

    # Place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()

    # can change to session states
    return gene_choice, chr_choice, ndd_genes[(ndd_genes['NAME']==st.session_state['ndd_gene_choice'])]
    

def rv_cohort_callback():
    st.session_state['old_rv_cohort_choice'] = st.session_state['rv_cohort_choice']
    st.session_state['rv_cohort_choice'] = st.session_state['new_rv_cohort_choice']

# rare variant method callback for session state
def method_callback():
    st.session_state['old_method_choice'] = st.session_state['method_choice']
    st.session_state['method_choice'] = st.session_state['new_method_choice']

# rare variant gene callback for session state
def rv_gene_callback():
    st.session_state['old_rv_gene_choice'] = st.session_state['rv_gene_choice']
    st.session_state['rv_gene_choice'] = st.session_state['new_rv_gene_choice']

# rare variant selector
def rv_select(rv_data):
    st.sidebar.markdown('### **Choose a cohort!**', unsafe_allow_html=True)

    # set cohort options
    rv_cohort_options = [i for i in rv_data['Study code'].unique()]

    # set default cohort
    if 'rv_cohort_choice' not in st.session_state:
        st.session_state['rv_cohort_choice'] = None
    if 'old_rv_cohort_choice' not in st.session_state:
        st.session_state['old_rv_cohort_choice'] = ""

    # dynamic change of cohort with selector
    st.session_state['rv_cohort_choice'] = st.sidebar.multiselect(label = 'Cohort Selection', label_visibility = 'collapsed', options=rv_cohort_options, default=st.session_state['rv_cohort_choice'], key='new_rv_cohort_choice', on_change=rv_cohort_callback)


    st.sidebar.markdown('### **Choose a discovery method!**', unsafe_allow_html=True)

    # set methods options
    method_options = [i for i in rv_data['Methods'].unique()]

    # set default methods
    if 'method_choice' not in st.session_state:
        st.session_state['method_choice'] = None
    if 'old_method_choice' not in st.session_state:
        st.session_state['old_method_choice'] = ""

    # dynamic change of methods with selector
    st.session_state['method_choice'] = st.sidebar.multiselect(label = 'Method Selection', label_visibility = 'collapsed', options=method_options, default=st.session_state['method_choice'], key='new_method_choice', on_change=method_callback)


    st.sidebar.markdown('### **Choose a gene!**', unsafe_allow_html=True)

    # set gene options
    rv_gene_options = [i for i in rv_data['Gene'].unique()]

    # set default gene
    if 'rv_gene_choice' not in st.session_state:
        st.session_state['rv_gene_choice'] = None
    if 'old_rv_gene_choice' not in st.session_state:
        st.session_state['old_rv_gene_choice'] = ""

    # dynamic change of gene with selector
    st.session_state['rv_gene_choice'] = st.sidebar.multiselect(label = 'Gene Selection', label_visibility = 'collapsed', options=rv_gene_options, default=st.session_state['rv_gene_choice'], key='new_rv_gene_choice', on_change=rv_gene_callback)

    # Place logos in sidebar
    st.sidebar.markdown('---')
    place_logos()