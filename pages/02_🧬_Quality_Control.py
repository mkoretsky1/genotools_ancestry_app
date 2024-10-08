import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from functools import reduce
from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select, release_select, config_page

config_page('Quality Control')

release_select()

# Pull data from different Google Cloud folders
gp2_data_bucket = get_gcloud_bucket('gp2tier2')

# Get qc metrics
qc_metrics_path = f'{st.session_state["release_bucket"]}/meta_data/qc_metrics/qc_metrics.csv'
df_qc = blob_as_csv(gp2_data_bucket, qc_metrics_path, sep = ',')  # current version: cannot split by cohort

# Gets master key (full GP2 release or selected cohort)
# gets master key (full GP2 release or selected cohort)
if st.session_state['release_choice'] == 8:
    master_key_path = f'{st.session_state["release_bucket"]}/clinical_data/master_key_release7_final.csv'
else:
    master_key_path = f'{st.session_state["release_bucket"]}/clinical_data/master_key_release{st.session_state["release_choice"]}_final.csv'
master_key = blob_as_csv(gp2_data_bucket, master_key_path, sep=',')
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
ordered_prune = ['insufficient_ancestry_sample_n','phenotype_not_reported', 'missing_idat', 'missing_bed', 'callrate_prune', 'sex_prune', 
                'het_prune', 'duplicated_prune']

# Get number of samples pruned at each step
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
    'missing_bed': 'Missing BED',
    'phenotype_not_reported': 'Phenotype Not Reported',
    'callrate_prune':'Call Rate Prune',
    'sex_prune': 'Sex Prune',
    'duplicated_prune': 'Duplicated',
    'het_prune': 'Heterozygosity Prune'
}

funnel_df.loc[:,'step_name'] = funnel_df.loc[:,'step'].map(steps_dict)  # prepares dataframe for funnel chart

# ancestry abbreviations and index encodings
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
            'FIN': 'Finnish',
            'CAH': 'Complex Admixture History'
        }

ancestry_index = {
            'AFR':3,
            'SAS':7,
            'EAS':8,
            'EUR':0,
            'AMR':2,
            'AJ':1,
            'AAC':4,
            'CAS':5,
            'MDE':6,
            'FIN':9,
            'CAH':10
        }

# remove CAS and MDE labels for releases 1 and 2, CAH for releases before 6
if st.session_state['release_choice'] < 3:
    for key in ['CAS','MDE']:
        ancestry_dict.pop(key)
        ancestry_index.pop(key)

if st.session_state['release_choice'] < 6:
    ancestry_dict.pop('CAH')
    ancestry_index.pop('CAH')

# Prepares dataframe for Relatedness Per Ancestry Plot
df_3 = master_key[(master_key['related'] == 1) | (master_key['pruned_reason'] == 'duplicated_prune')]
df_3 = df_3[['label','pruned']]

df_4 = pd.DataFrame()

# if len(df_3) > 0:
df_4_dicts = [] 

# Get related and duplicated counts by ancestry
for label in ancestry_dict:
    ancestry_df_dict = {}
    if label in df_3['label'].unique():
        df_3_ancestry = df_3[df_3['label'] == label]
        ancestry_df_dict['ancestry'] = label
        ancestry_df_dict['related_count'] = df_3_ancestry[df_3_ancestry['pruned'] == 0].shape[0]
        ancestry_df_dict['duplicated_count'] = df_3_ancestry[df_3_ancestry['pruned'] == 1].shape[0]
    else:
        ancestry_df_dict['ancestry'] = label
        ancestry_df_dict['related_count'] = 0
        ancestry_df_dict['duplicated_count'] = 0

    df_4_dicts.append(ancestry_df_dict)

df_4 = pd.DataFrame(df_4_dicts)
# df_4 = df_4.sort_values(by=['related_count', 'duplicated_count'], ascending=False)
df_4.loc[:,'label'] = df_4.loc[:,'ancestry'].map(ancestry_dict)
df_4.loc[:, 'label_index'] = df_4.loc[:,'ancestry'].map(ancestry_index)
df_4 = df_4.sort_values(by=['label_index'], ascending=True)
df_4.set_index('ancestry', inplace=True)


###### Variant pruning

# Same variant pruning counts for all cohorts
if st.session_state['release_choice'] >= 6:
    df_5 = df_qc
else:
    df_5 = df_qc.query("step == 'variant_prune'")
df_5 = df_5[['ancestry', 'pruned_count', 'metric']]

# Counts per each variant filtering category in qc_metrics.csv
df_5_geno = df_5.query("metric == 'geno_removed_count'").reset_index(drop=True)
df_5_geno = df_5_geno.rename(columns={'pruned_count': 'geno_removed_count'})
df_5_geno = df_5_geno.drop(columns=['metric'], axis=1)

df_5_mis = df_5.query("metric == 'mis_removed_count'").reset_index(drop=True)
df_5_mis = df_5_mis.rename(columns={'pruned_count': 'mis_removed_count'})
df_5_mis = df_5_mis.drop(columns=['metric'], axis=1)

df_5_haplo = df_5.query("metric == 'haplotype_removed_count'").reset_index(drop=True)
df_5_haplo = df_5_haplo.rename(columns={'pruned_count': 'haplotype_removed_count'})
df_5_haplo = df_5_haplo.drop(columns=['metric'], axis=1)

df_5_hwe = df_5.query("metric == 'hwe_removed_count'").reset_index(drop=True)
df_5_hwe = df_5_hwe.rename(columns={'pruned_count': 'hwe_removed_count'})
df_5_hwe = df_5_hwe.drop(columns=['metric'], axis=1)

df_5_total = df_5.query("metric == 'total_removed_count'").reset_index(drop=True)
df_5_total = df_5_total.rename(columns={'pruned_count': 'total_removed_count'})
df_5_total = df_5_total.drop(columns=['metric'], axis=1)

data = [df_5_geno, df_5_mis, df_5_haplo, df_5_hwe, df_5_total]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['ancestry'], how='outer'), data)
df_merged.set_index('ancestry', inplace=True)

df_6 = df_qc.loc[df_qc['pass'] == False]
df_6 = df_6.reset_index(drop=True)


###### Create plots for prepared dataframes

# Pruning Steps plot
funnel_counts = go.Figure(go.Funnelarea(
    text = [f'<b>{i}</b>' for i in funnel_df['step_name']],
    values = funnel_df['remaining_samples'],
    marker = {"colors": ["#999999", "#E69F00", "#56B4E9", "#009E73", "#AA4499", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]},
    opacity = 1.0, textinfo = 'text',
    customdata=funnel_df['remaining_samples'],
    hovertemplate = 'Remaining Samples:' + '<br>%{customdata[0]:.f}'+'<extra></extra>'))

funnel_counts.update_layout(showlegend=False, margin=dict(l=0, r=300, t=10, b=0))

# Relatedness Per Anceestry Plot
if len(df_4) > 0:
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
                ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness using KING, where a  cutoff of 0.0884 \
            was used to determine second degree relatedness and 0.354 is used to determine duplicates. For purposes of imputation,\
                related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity where F <= -0.25 of\
                F>= 0.25.')

left_col1, right_col1 = st.columns([1.5,2])

with left_col1:
    st.header("**All Sample Filtering Counts**")
    st.plotly_chart(funnel_counts)
    
with right_col1:
    if len(df_4) > 0:
        # if st.session_state['cohort_choice'] == f'GP2 Release 3 FULL':  # will disappear if other cohort selected
        #     st.header("**Relatedness per Ancestry**")
        #     st.plotly_chart(bar_3)
        # if st.session_state['release_choice'] == 4:
        st.header("**Relatedness per Ancestry**")
        st.plotly_chart(bar_3)

st.markdown('---')

st.header('QC Step 2: Variant-Level Filtering')
var_exp = st.expander("Description", expanded=False)
with var_exp:
    st.markdown('Variants are pruned for missingness by case-control where P<=1e-4 to detect platform/batch differences in case-control status.\
                Next, variants are pruned for missingness by haplotype for flanking variants where P<=1e-4. Lastly, controls are filtered for HWE \
                at a threshold of 1e-4. Please note that for each release, variant pruning is performed in an ancestry-specific manner, and thus \
                the numbers in the bar chart below will not change based on cohort selection within the same release.')

st.header("**Variant Filtering per Ancestry**")
st.plotly_chart(bar_6)
st.markdown('---')

if df_6.shape[0]>0:
    st.markdown("**Failed Prune Steps**")
    failed_prune_exp = st.expander("Description", expanded=False)
    with failed_prune_exp:
        st.write('Prune step considered "failed" if there was an insufficient number of samples within an ancestry to complete the \
                step, even if no samples were pruned.')
        
    hide_table_row_index = """<style>thead tr th:first-child {display:none} tbody th {display:none}"""
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df_6)





