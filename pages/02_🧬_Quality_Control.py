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

config_page('Quality Control')

# Pull data from different Google Cloud folders
gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
df_qc = blob_as_csv(gp2_sample_bucket, 'qc_metrics.csv', sep = ',')  # current version: cannot split by cohort

# Gets master key (full GP2 release or selected cohort)
master_key = blob_as_csv(gp2_sample_bucket, f'master_key_release3_final.csv', sep=',')
cohort_select(master_key)

# Necessary dataframes for QC Plots
master_key = st.session_state['master_key']  # plots full GP2 release metrics by default
st.session_state['df_qc'] = df_qc
st.session_state['pre_sample_n'] = master_key['GP2sampleID'].count()
st.session_state['remaining_n'] = master_key['GP2sampleID'].count()

###### All-sample pruning

pre_QC_total = master_key['GP2sampleID'].count()
funnel_df = pd.DataFrame(columns=['remaining_samples', 'step'])
funnel_df.loc[0] = pd.Series({'remaining_samples':pre_QC_total, 'step':'pre_QC'})

hold_prunes = master_key['pruned_reason'].value_counts().rename_axis('pruned_reason').reset_index(name='pruned_counts')
remaining_samples = pre_QC_total

# Proper order of pruning steps to pull from Master Key
ordered_prune = ['insufficient_ancestry_sample_n','phenotype_not_reported', 'missing_idat', 'corrupted_idat', 'callrate_prune', 'sex_prune', 
                'het_prune', 'duplicated']

for prunes in ordered_prune:
    step_name = prunes
    obs_pruned = hold_prunes['pruned_reason'].tolist()
    if prunes not in obs_pruned:
        remaining_samples -= 0
    else:
        row = hold_prunes.loc[hold_prunes['pruned_reason'] == prunes]
        remaining_samples -= row.iloc[0]['pruned_counts']
    funnel_df.loc[len(funnel_df.index)] = pd.Series({'remaining_samples':remaining_samples, 'step':step_name})

# Full names of pruning steps to map to abbreviations in Master Key
steps_dict = {
    'pre_QC': 'Pre-QC',
    'insufficient_ancestry_sample_n': 'Insufficient Ancestry Count',
    'missing_idat': 'Missing IDAT',
    'corrupted_idat': 'Corrupted IDAT',
    'phenotype_not_reported': 'Phenotype Not Reported',
    'callrate_prune':'Call Rate Prune',
    'sex_prune': 'Sex Prune',
    'duplicated': 'Duplicated',
    'het_prune': 'Heterozygosity Prune'
}

funnel_df.loc[:,'step_name'] = funnel_df.loc[:,'step'].map(steps_dict)  # prepares dataframe for funnel chart

# Prepares dataframe for Relatedness Per Ancestry Plot (only for full GP2 Release, not any other selected cohorts)
df_3 = df_qc.query("step == 'related_prune'")
df_3 = df_3[['ancestry', 'pruned_count', 'metric']]

# Counts all samples marked for related prune
df_3_related = df_3.query("metric == 'related_count'").reset_index(drop=True)
df_3_related = df_3_related.rename(columns={'pruned_count': 'related_count'})
df_3_related = df_3_related.drop('metric', 1)

# Counts samples labeled as "duplicated" within the related prune labels
df_3_duplicated = df_3.query("metric == 'duplicated_count'").reset_index(drop=True)
df_3_duplicated = df_3_duplicated.rename(columns={'pruned_count': 'duplicated_count'})
df_3_duplicated = df_3_duplicated.drop('metric', 1)

# Full names of ancestry labels to map abbreviations in Master Key
df_4 = pd.merge(df_3_related, df_3_duplicated, on="ancestry")
ancestry_dict = {
        'AFR':'African',
        'SAS':'South Asian',
        'EAS':'East Asian',
        'EUR':'European',
        'AMR': 'American',
        'AJ': 'Ashkenazi Jewish',
        'AAC': 'African American/Afro-Caribbean',
        'CAS': 'Central Asian',
        'MDE': 'Middle Eastern',
        'FIN': 'Finnish'
    }

df_4.loc[:,'label'] = df_4.loc[:,'ancestry'].map(ancestry_dict)
df_4.set_index('ancestry', inplace=True)


###### Variant pruning

# Same variant pruning counts for all cohorts 
df_5 = df_qc.query("step == 'variant_prune'")
df_5 = df_5[['ancestry', 'pruned_count', 'metric']]

# Counts per each variant filtering category in qc_metrics.csv
df_5_geno = df_5.query("metric == 'geno_removed_count'").reset_index(drop=True)
df_5_geno = df_5_geno.rename(columns={'pruned_count': 'geno_removed_count'})
df_5_geno = df_5_geno.drop('metric', 1)

df_5_mis = df_5.query("metric == 'mis_removed_count'").reset_index(drop=True)
df_5_mis = df_5_mis.rename(columns={'pruned_count': 'mis_removed_count'})
df_5_mis = df_5_mis.drop('metric', 1)

df_5_haplo = df_5.query("metric == 'haplotype_removed_count'").reset_index(drop=True)
df_5_haplo = df_5_haplo.rename(columns={'pruned_count': 'haplotype_removed_count'})
df_5_haplo = df_5_haplo.drop('metric', 1)

df_5_hwe = df_5.query("metric == 'hwe_removed_count'").reset_index(drop=True)
df_5_hwe = df_5_hwe.rename(columns={'pruned_count': 'hwe_removed_count'})
df_5_hwe = df_5_hwe.drop('metric', 1)

df_5_total = df_5.query("metric == 'total_removed_count'").reset_index(drop=True)
df_5_total = df_5_total.rename(columns={'pruned_count': 'total_removed_count'})
df_5_total = df_5_total.drop('metric', 1)

data = [df_5_geno, df_5_mis, df_5_haplo, df_5_hwe, df_5_total]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['ancestry'], how='outer'), data)
df_merged.set_index('ancestry', inplace=True)

df_6 = df_qc.loc[df_qc['pass'] == False]
df_6 = df_6.reset_index(drop=True)


###### Create plots for prepared dataframes

# Pruning Steps plot
funnel_counts = go.Figure(go.Funnelarea(
    text = funnel_df['step_name'],
    values = funnel_df['remaining_samples'],
    marker = {"colors": ["#999999", "#E69F00", "#56B4E9", "#009E73", "#AA4499", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]},
    opacity = 0.9, textinfo = 'text',
    customdata=funnel_df['remaining_samples'],
    hovertemplate = 'Remaining Samples:' + '<br>%{customdata[0]:.f}'+'<extra></extra>'))

funnel_counts.update_layout(showlegend = False, margin=dict(l=0, r=300, t=10, b=10))

# Relatedness Per Anceestry Plot
bar_3 = go.Figure(data=[
        go.Bar(y=df_4.label, x=df_4['related_count'], orientation='h', name="Related", base=0, marker_color="#0072B2"),
        go.Bar(y=df_4.label, x=-df_4['duplicated_count'], orientation='h', name="Duplicated", base=0, marker_color="#D55E00")])

bar_3.update_layout(barmode='stack')

bar_3.update_yaxes(
    ticktext=df_4.label,
    tickvals=df_4.label
)

bar_3.update_layout(
    autosize=False,
    height=500, width = 750
)

bar_3.update_layout(
    margin=dict(l=0, r=200, t=10, b=60),
)

# Variant-Level Pruning plot
bar_6 = go.Figure(go.Bar(x=df_merged.index, y=df_merged['geno_removed_count'], name='Geno Removed Count', marker_color = "#0072B2"))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['mis_removed_count'], name='Mis Removed Count', marker_color = "#882255"))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['haplotype_removed_count'], name='Haplotype Removed Count', marker_color = "#44AA99"))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['hwe_removed_count'], name='HWE Removed Count', marker_color = "#D55E00"))

bar_6.update_layout(
    xaxis=dict(
        categoryorder='total descending',
        title='Ancestry',
        tickfont_size=14),
        yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14),
        barmode='stack', 
        width=1100, height=600)


###### Create App

st.title(f'{st.session_state["cohort_choice"]} Metrics')

st.header('QC Step 1: Sample-Level Filtering')
sample_exp = st.expander("Description", expanded=False)
with sample_exp:
    st.markdown('Genotypes are pruned for call rate with maximum sample genotype missingness of 0.02 (--mind 0.02). Samples which pass\
                call rate pruning are then pruned for discordant sex where samples with 0.25 <= sex F <= 0.75 are pruned. Sex F < 0.25\
                are female and Sex F > 0.75 are male. Samples that pass sex pruning are then differentiated by ancestry (refer to\
                ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness where genetic relatedness matrix (grm)\
                cutoff 0.125 is used to determine second-degree relatedness and 0.95 is used to determine duplicates. For purposes of imputation,\
                related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity where F <= -0.25 of\
                F>= 0.25.')

left_col1, right_col1 = st.columns([1.5,2])

with left_col1:
    st.header("**All Sample Filtering Counts**")
    st.plotly_chart(funnel_counts)
    
with right_col1:
    # st.dataframe(funnel_df[['step_name', 'remaining_samples']].rename(columns = {'step_name': 'QC Step', 'remaining_samples': 'Remaining Samples'}))
    if st.session_state['cohort_choice'] == 'GP2 Release 3 FULL':  # will disappear if other cohort selected
        st.header("**Relatedness per Ancestry**")
        st.plotly_chart(bar_3)

st.markdown('---')

st.header('QC Step 2: Variant-Level Filtering')
var_exp = st.expander("Description", expanded=False)
with var_exp:
    st.markdown('Variants are pruned for missingness by case-control where P<=1e-4 to detect platform/batch differences in case-control status.\
                Next, variants are pruned for missingness by haplotype for flanking variants where P<=1e-4. Lastly, controls are filtered for HWE\
                at a threshold of 1e-4.')

st.header("**Variant Filtering per Ancestry**")
st.plotly_chart(bar_6)
st.markdown('---')

if df_6.shape[0]>0:
    st.markdown("**Failed Prune Steps**")
    failed_prune_exp = st.expander("Description", expanded=False)
    with failed_prune_exp:
        st.write('Prune step considered "failed" if there was an insufficient number of samples within an ancestry to complete the \
                step, even if no sampels were pruned.')
    st.table(df_6)





