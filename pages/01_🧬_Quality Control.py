import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from PIL import Image
import datetime
from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select
from functools import reduce

st.set_page_config(page_title = "Quality Control", layout = 'wide')

tabFull, tabCohort = st.tabs(["Fulll GP2 Release 3", "Selected Cohort"])

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)
df_qc = blob_as_csv(gp2_sample_bucket, 'qc_metrics.csv', sep = ',')

master_key = blob_as_csv(gp2_sample_bucket, f'master_key_release3_final.csv', sep=',')
cohort_select(master_key)

def createQC(master_key, data_name, rel_plot = True): 
    st.session_state['df_qc'] = df_qc
    st.session_state['pre_sample_n'] = master_key['GP2sampleID'].count()
    st.session_state['remaining_n'] = master_key['GP2sampleID'].count()

    ###### All-sample pruning

    pre_QC_total = master_key['GP2sampleID'].count()
    funnel_df = pd.DataFrame(columns=['remaining_samples', 'step'])
    funnel_df.loc[0] = pd.Series({'remaining_samples':pre_QC_total, 'step':'pre_QC'})
    # st.dataframe(funnel_df.head())

    hold_prunes = master_key['pruned_reason'].value_counts().rename_axis('pruned_reason').reset_index(name='pruned_counts')
    # st.dataframe(hold_prunes)

    remaining_samples = pre_QC_total

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

    # funnel_df = pd.DataFrame({'remaining_samples':remain_samples_x,'step':steps_y})
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

    funnel_df.loc[:,'step_name'] = funnel_df.loc[:,'step'].map(steps_dict)

    # fig_percents = px.funnel_area(names=funnel_df['step_name'],
    #                     values=funnel_df['remaining_samples'])
    # st.plotly_chart(fig_percents)

    # ugly_funnel = go.Figure(go.Funnel(
    #     y = funnel_df['step_name'],
    #     x = funnel_df['remaining_samples']))
    # st.plotly_chart(ugly_funnel)

    # funnel_counts = go.Figure(go.Funnelarea(
    #     text = funnel_df['step_name'],
    #     values = funnel_df['remaining_samples'], textinfo = "value" 
    #     ))
    # st.plotly_chart(funnel_counts)

    df_3 = df_qc.query("step == 'related_prune'")
    df_3 = df_3[['ancestry', 'pruned_count', 'metric']]

    df_3_related = df_3.query("metric == 'related_count'").reset_index(drop=True)
    df_3_related = df_3_related.rename(columns={'pruned_count': 'related_count'})
    df_3_related = df_3_related.drop('metric', 1)
    # st.dataframe(df_3_related.head())

    df_3_duplicated = df_3.query("metric == 'duplicated_count'").reset_index(drop=True)
    df_3_duplicated = df_3_duplicated.rename(columns={'pruned_count': 'duplicated_count'})
    df_3_duplicated = df_3_duplicated.drop('metric', 1)
    # st.dataframe(df_3_duplicated.head())

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
    # st.dataframe(df_4)

    ### Re-write Relatedness Per Ancestry Graph using only Master Key
    # df_3_related = master_key.query("pruned_reason == 'related_prune'")
    # df_3_related = df_3_related[['label', 'pruned_reason']]

    # df_3_duplicated = master_key.query("pruned_reason == 'duplicated'")
    # df_3_duplicated = df_3_duplicated[['label', 'pruned_reason']]

    # st.dataframe(df_3_related.head())
    # st.dataframe(df_3_duplicated.head())


        
    ###### Variant pruning
    df_5 = df_qc.query("step == 'variant_prune'")
    df_5 = df_5[['ancestry', 'pruned_count', 'metric']]

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


    ###### Plotting 
    funnel_counts = go.Figure(go.Funnelarea(
        text = funnel_df['step_name'],
        values = funnel_df['remaining_samples'], textinfo = "text"))

    funnel_counts.update_layout(showlegend = False, margin=dict(l=0, r=0, t=10, b=60))
    # funnel_counts.update_traces(title = "Hi", showlegend = False)


    #customize figure
    # ugly_funnel.update_layout(
    #         margin=dict(l=0, r=0, t=60, b=80),
    #         yaxis_title="QC Step"
    #     )

    bar_3 = go.Figure(data=[
            go.Bar(y=df_4.label, x=df_4['related_count'], orientation='h', name="Related", base=0),
            go.Bar(y=df_4.label, x=-df_4['duplicated_count'], orientation='h', name="Duplicated", base=0)])

    bar_3.update_layout(barmode='stack')

    bar_3.update_yaxes(
        ticktext=df_4.label,
        tickvals=df_4.label
    )

    bar_3.update_layout(
        autosize=False,
        height=600, width = 1000
    )

    bar_3.update_layout(
        margin=dict(l=0, r=0, t=60, b=80),
    )

    bar_6 = go.Figure(go.Bar(x=df_merged.index, y=df_merged['geno_removed_count'], name='Geno Removed Count'))
    bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['mis_removed_count'], name='Mis Removed Count'))
    bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['haplotype_removed_count'], name='Haplotype Removed Count'))
    bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['hwe_removed_count'], name='Hwe Removed Count'))

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
    # if 'cohort_choice' in st.session_state:
    #     st.title(f'{st.session_state["cohort_choice"]} Quality Control Metrics')   
    # else:
    #     st.title('**Full GP2 Cohort Quality Control Metrics**')

    st.title(data_name)

    st.header('QC Step 1: Sample Filtering')
    sample_exp = st.expander("Description", expanded=False)
    with sample_exp:
            st.write('\
                    1. Call Rate: Missingness per-individual > 0.02  \n\
                    2. Sex: F-coefficient < 0.25 is Female, F-coefficient > 0.75 is Male. 0.25 >= F <= 0.75 are outliers  \n\
                    3. Heterozygosity: Keep samples with -0.25 < F-coefficient < 0.25  \n\
                    4. Relatedness: Flag Samples with relatedness > 0.125 as cousins or more closely-related, Prune relatedness > 0.95 as Duplicates\
                ')

    st.header("**All Sample Filtering Counts**")
    left_col1, right_col1 = st.columns([1,2])

    with right_col1:
        st.plotly_chart(funnel_counts)
        
    with left_col1:
        st.markdown('### ')
        st.dataframe(funnel_df[['step_name', 'remaining_samples']].rename(columns = {'step_name': 'QC Step', 'remaining_samples': 'Remaining Samples'}))

    # funnel2_left, funnel2_right = st.columns([3,2])
    # with funnel2_left:
    #     st.plotly_chart(fig_percents)

    # with funnel2_right:
    #     st.dataframe(funnel_df[['step_name', 'remaining_samples']].rename(columns = {'step_name': 'QC Step', 'remaining_samples': 'Remaining Samples'}))


    # st.plotly_chart(ugly_funnel)

    if rel_plot:
        st.header("**Relatedness per Ancestry**")
        st.plotly_chart(bar_3)
    st.markdown('---')
                
    # left_col2, right_col2 = st.columns([1.25,1])

    st.header('QC Step 2: Variant Filtering')
    var_exp = st.expander("Description", expanded=False)
    with var_exp:
        st.write('\
                1. Genotype Missingness <= 0.05  \n\
                2. Case/Control Missingness: P > 1e-4  \n\
                3. Haplotype Missingness: P > 1e-4  \n\
                4. Hardy-Weinberg Equilibrium: P > 1e-4  \n\
            ')

    st.header("**Variant Filtering per Ancestry**")
    st.plotly_chart(bar_6)
    st.markdown('---')

    if df_6.shape[0]>0:
        st.markdown("**Failed Prune Steps**")
        failed_prune_exp = st.expander("Description", expanded=False)
        with failed_prune_exp:
            st.write('Prune step considered "failed" if there was an insufficient number of samples within an ancestry to complete the step.')
        st.table(df_6)

with tabFull:
    master_key = blob_as_csv(gp2_sample_bucket, 'master_key_release3_final.csv', sep = ',')
    data_name = 'Full GP2 3 Release Quality Control Metrics'

    createQC(master_key, data_name)

with tabCohort:
    if 'cohort_choice' in st.session_state:
        if st.session_state['cohort_choice'] == 'GP2 Release 3 FULL':
            st.info('Use the drop-down menu on the sidebar to display QC metrics for a different cohort')
        else:
            if 'master_key' in st.session_state:
                master_key2 = st.session_state['master_key']
                data_name = f'{st.session_state["cohort_choice"]} Quality Control Metrics'

                createQC(master_key2, data_name, rel_plot = False)
    else:
        st.error('Use the drop-down menu on the sidebar to display QC metrics for a specific cohort')



