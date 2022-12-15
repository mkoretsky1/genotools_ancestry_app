import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from functools import reduce
from PIL import Image

from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select, config_page

config_page('Metadata')

st.title('GP2 Release 3 Metadata')

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
df_qc = blob_as_csv(gp2_sample_bucket, 'qc_metrics.csv', sep = ',')  # current version: cannot split by cohort

# Gets master key (full GP2 release or selected cohort)
master_key = blob_as_csv(gp2_sample_bucket, f'master_key_release3_final.csv', sep=',')
cohort_select(master_key)

master_key = st.session_state['master_key']  # plots full GP2 release metrics by default
master_key.rename(columns = {'age': 'Age', 'sex_for_qc': 'Sex'}, inplace = True)

master_key['Sex'].replace(1, 'Male', inplace = True)
master_key['Sex'].replace(2, 'Female', inplace = True)
master_key['Sex'].replace(0, 'Unknown', inplace = True)

st.dataframe(master_key.head())

fig = px.histogram(master_key['Age'], x = 'Age', nbins = 25)
st.plotly_chart(fig)

# create the bins
# min_age = int(min(age))
# max_age = int(max(age))
# counts, bins = np.histogram(age, bins=range(min_age, max_age, 5))
# bins = 0.5 * (bins[:-1] + bins[1:])
# fig = px.bar(x=bins, y=counts, labels={'x':'Age', 'y':'Count'})
# st.plotly_chart(fig)

st.markdown('#### Stratify Age by:')
sex = st.checkbox('Sex')
phenotype= st.checkbox('Phenotype')

if sex:
    fig = px.histogram(master_key, x="Age", color="Sex", nbins = 25)
    st.plotly_chart(fig)
if phenotype:
    fig = px.histogram(master_key, x="Age", color="Phenotype", nbins = 25)
    st.plotly_chart(fig)

male_pheno = master_key.loc[master_key['Sex'] == 'Male', 'Phenotype']
female_pheno = master_key.loc[master_key['Sex'] == 'Female', 'Phenotype']

combined_counts = pd.DataFrame()
combined_counts['Male'] = male_pheno.value_counts()
combined_counts['Female'] = female_pheno.value_counts()

st.dataframe(combined_counts.transpose())
