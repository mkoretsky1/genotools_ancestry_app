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
from hold_data import blob_as_csv, get_gcloud_bucket
from functools import reduce

st.set_page_config(page_title = "Quality Control", layout = 'wide')

gp2_sample_bucket_name = 'gp2_sample_data'
gp2_sample_bucket = get_gcloud_bucket(gp2_sample_bucket_name)

master_key = blob_as_csv(gp2_sample_bucket, 'master_key_release3_final.csv', sep = ',')
# st.dataframe(master_key.head())

df_qc = blob_as_csv(gp2_sample_bucket, 'qc_metrics.csv', sep = ',')

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
for i in range(len(hold_prunes)):
    step_name = hold_prunes['pruned_reason'].iloc[i]
    remaining_samples -= hold_prunes['pruned_counts'].iloc[i]
    funnel_df.loc[i+1] = pd.Series({'remaining_samples':remaining_samples, 'step':step_name})

if 'het_prune' not in funnel_df['step']:
    funnel_df.loc[i+2] = pd.Series({'remaining_samples':remaining_samples, 'step':'het_prune'})
# st.dataframe(funnel_df)

# funnel_df = pd.DataFrame({'remaining_samples':remain_samples_x,'step':steps_y})
steps_dict = {
    'pre_QC': 'Pre-QC',
    'missing_idat': 'Missing IDAT',
    'corrupted_idat': 'Corrupted IDAT',
    'duplicated': 'Duplicated',
    'phenotype_not_reported': 'Phenotype Not Reported',
    'callrate_prune':'Call Rate Prune',
    'sex_prune': 'Sex Prune',
    'related_prune': 'Relatedness Prune',
    'het_prune': 'Heterozygosity Prune'
}

funnel_df.loc[:,'step_name'] = funnel_df.loc[:,'step'].map(steps_dict)

fig_percents = px.funnel_area(names=funnel_df['step_name'],
                    values=funnel_df['remaining_samples'])
# st.plotly_chart(fig_percents)

ugly_funnel = go.Figure(go.Funnel(
    y = funnel_df['step_name'],
    x = funnel_df['remaining_samples']))
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

df_3_duplicated = df_3.query("metric == 'duplicated_count'").reset_index(drop=True)
df_3_duplicated = df_3_duplicated.rename(columns={'pruned_count': 'duplicated_count'})
df_3_duplicated = df_3_duplicated.drop('metric', 1)

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
    values = funnel_df['remaining_samples'], textinfo = "text"
    ))

funnel_counts.update_traces(showlegend = False)
# funnel_counts.update_layout(
#         margin=dict(l=0, r=0, t=60, b=80),
#         yaxis_title=funnel_df['step_name'])

#customize figure
ugly_funnel.update_layout(
        margin=dict(l=0, r=0, t=60, b=80),
        yaxis_title="QC Step"
    )

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
    height=600, width=700
)

bar_3.update_layout(
    margin=dict(l=0, r=0, t=60, b=80),
)

bar_6 = go.Figure(go.Bar(x=df_merged.index, y=df_merged['geno_removed_count'], name='Geno Removed Count'))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['mis_removed_count'], name='Mis Removed Count'))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['haplotype_removed_count'], name='Haplotype Removed Count'))
bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['hwe_removed_count'], name='Hwe removed Count'))

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
        width=1200, height=600)

###### Create App
st.title('**GenoTools Quality Control**')
            
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
left_col1, right_col1 = st.columns([2,2])

with left_col1:
    st.plotly_chart(funnel_counts)

with right_col1:
    st.dataframe(funnel_df[['step_name', 'remaining_samples']].rename(columns = {'step_name': 'QC Step', 'remaining_samples': 'Remaining Samples'}))

funnel2_left, funnel2_right = st.columns([3,2])
with funnel2_left:
    st.plotly_chart(fig_percents)

with funnel2_right:
    st.dataframe(funnel_df[['step_name', 'remaining_samples']].rename(columns = {'step_name': 'QC Step', 'remaining_samples': 'Remaining Samples'}))


# st.plotly_chart(ugly_funnel)


st.header("**Relatedness per Ancestry**")
st.plotly_chart(bar_3)
st.markdown('---')
            
eft_col2, right_col2 = st.columns([1.25,1])

st.header('QC Step 3: Variant Filtering')
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
    st.table(df_6)
    st.markdown('---')


######## Dan's Code
# if 'df_qc' in st.session_state:
#         df_qc = st.session_state['df_qc']
            
#         # pre process
#         # all sample prune
#         df_2 = df_qc.query("level == 'sample'")
#         df_2['sum'] = df_2.groupby('step')['pruned_count'].transform('sum')
#         df_2 = df_2[['step','sum']]
#         df_2 = df_2.drop_duplicates(subset=['step', 'sum'], keep='first')
#         remaining_list = []

#         start_n = st.session_state['remaining_n']
#         for n in df_2['sum']:

#             st.session_state['remaining_n']=st.session_state['remaining_n']-n
#             rem = st.session_state['remaining_n']
#             remaining_list.append(rem)


#         remain_samples_x = [start_n] + remaining_list
#         steps_y = ['pre_QC'] + list(df_2.step)
#         funnel_df = pd.DataFrame({'remaining_samples':remain_samples_x,'step':steps_y})
#         steps_dict = {
#             'pre_QC': 'Pre-QC',
#             'callrate_prune':'Call Rate Prune',
#             'sex_prune': 'Sex Prune',
#             'het_prune': 'Heterozygosity Prune',
#             'related_prune': 'Relatedness Prune'
#             }
#         # funnel_df.loc[:,'step_name'] = funnel_df.replace({"step": steps_dict})
#         funnel_df.loc[:,'step_name'] = funnel_df.loc[:,'step'].map(steps_dict)

#         df_3 = df_qc.query("step == 'related_prune'")
#         df_3 = df_3[['ancestry', 'pruned_count', 'metric']]

#         df_3_related = df_3.query("metric == 'related_count'").reset_index(drop=True)
#         df_3_related = df_3_related.rename(columns={'pruned_count': 'related_count'})
#         df_3_related = df_3_related.drop('metric', 1)

#         df_3_duplicated = df_3.query("metric == 'duplicated_count'").reset_index(drop=True)
#         df_3_duplicated = df_3_duplicated.rename(columns={'pruned_count': 'duplicated_count'})
#         df_3_duplicated = df_3_duplicated.drop('metric', 1)

#         df_4 = pd.merge(df_3_related, df_3_duplicated, on="ancestry")
#         ancestry_dict = {
#             'FIN': 'Finnish (FIN)',
#             'EAS': 'East Asian (EAS)',
#             'AAC': 'African Admixted/Carribbean (AAC)',
#             'AJ': 'Ashkenazi (AJ)',
#             'SAS': 'South Asian (SAS)',
#             'AMR': 'Native American/Latino (AMR)',
#             'EUR': 'European (EUR)'
#         }

#         df_4.loc[:,'label'] = df_4.loc[:,'ancestry'].map(ancestry_dict)
#         df_4.set_index('ancestry', inplace=True)

       
#         #variant prune
#         df_5 = df_qc.query("step == 'variant_prune'")
#         df_5 = df_5[['ancestry', 'pruned_count', 'metric']]

#         df_5_geno = df_5.query("metric == 'geno_removed_count'").reset_index(drop=True)
#         df_5_geno = df_5_geno.rename(columns={'pruned_count': 'geno_removed_count'})
#         df_5_geno = df_5_geno.drop('metric', 1)

#         df_5_mis = df_5.query("metric == 'mis_removed_count'").reset_index(drop=True)
#         df_5_mis = df_5_mis.rename(columns={'pruned_count': 'mis_removed_count'})
#         df_5_mis = df_5_mis.drop('metric', 1)

#         df_5_haplo = df_5.query("metric == 'haplotype_removed_count'").reset_index(drop=True)
#         df_5_haplo = df_5_haplo.rename(columns={'pruned_count': 'haplotype_removed_count'})
#         df_5_haplo = df_5_haplo.drop('metric', 1)

#         df_5_hwe = df_5.query("metric == 'hwe_removed_count'").reset_index(drop=True)
#         df_5_hwe = df_5_hwe.rename(columns={'pruned_count': 'hwe_removed_count'})
#         df_5_hwe = df_5_hwe.drop('metric', 1)

#         df_5_total = df_5.query("metric == 'total_removed_count'").reset_index(drop=True)
#         df_5_total = df_5_total.rename(columns={'pruned_count': 'total_removed_count'})
#         df_5_total = df_5_total.drop('metric', 1)

#         data = [df_5_geno, df_5_mis, df_5_haplo, df_5_hwe, df_5_total]
#         df_merged = reduce(lambda left,right: pd.merge(left,right,on=['ancestry'], how='outer'), data)
#         df_merged.set_index('ancestry', inplace=True)

#         df_6 = df_qc.loc[df_qc['pass'] == False]
#         df_6 = df_6.reset_index(drop=True)


#         # plotting
#         #simple bar chart for callrate_prune and sex_prune
#         funnel = px.funnel(
#             funnel_df, 
#             x='remaining_samples', 
#             y='step_name', 
#             height=600, width=550
#             )

#         #customize figure
#         funnel.update_traces(
#             marker_line_width=1.5, 
#             opacity=0.8
#             )
#         funnel.update_layout(
#         margin=dict(l=0, r=0, t=60, b=80),
#         yaxis_title="QC Step"
#     )
#             # marker_color='rgb(158,202,225)', 
#             # marker_line_color='rgb(8,48,107)',
#         bar_3 = go.Figure(
#             data=[
#                 go.Bar(y=df_4.label, x=df_4['related_count'], orientation='h', name="Related", base=0),
#                 go.Bar(y=df_4.label, x=-df_4['duplicated_count'], orientation='h', name="Duplicated", base=0)
#                 ])

#         bar_3.update_layout(barmode='stack')

#         bar_3.update_yaxes(
#             ticktext=df_4.label,
#             tickvals=df_4.label
#         )

#         bar_3.update_layout(
#             autosize=False,
#             height=600, width=700
#         )

#         bar_3.update_layout(
#         margin=dict(l=0, r=0, t=60, b=80),
#     )

#         bar_6 = go.Figure(go.Bar(x=df_merged.index, y=df_merged['geno_removed_count'], name='Geno Removed Count'))
#         bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['mis_removed_count'], name='Mis Removed Count'))
#         bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['haplotype_removed_count'], name='Haplotype Removed Count'))
#         bar_6.add_trace(go.Bar(x=df_merged.index, y=df_merged['hwe_removed_count'], name='Hwe removed Count'))

#         bar_6.update_layout(
#             xaxis=dict(
#                 categoryorder='total descending',
#                 title='Ancestry',
#                 tickfont_size=14,
                
#             ),
#             yaxis=dict(
#                 title='Count',
#                 titlefont_size=16,
#                 tickfont_size=14,
#             ),

#             barmode='stack', 
#             width=1200, height=600
#         )


       

#         # create app
#         if 'remaining_n' in st.session_state:

#             st.title('**GenoTools Quality Control**')
            
#             st.header('QC Step 1: Sample Filtering')
#             sample_exp = st.expander("Description", expanded=False)
#             with sample_exp:
#                 st.write('\
#                         1. Call Rate: Missingness per-individual > 0.02  \n\
#                         2. Sex: F-coefficient < 0.25 is Female, F-coefficient > 0.75 is Male. 0.25 >= F <= 0.75 are outliers  \n\
#                         3. Heterozygosity: Keep samples with -0.25 < F-coefficient < 0.25  \n\
#                         4. Relatedness: Flag Samples with relatedness > 0.125 as cousins or more closely-related, Prune relatedness > 0.95 as Duplicates\
#                 ')

#             left_col1, right_col1 = st.columns([1,1])
#             with left_col1:
#                 st.header("**All Sample Filtering Counts**")
#                 st.plotly_chart(funnel)

#             with right_col1:
#                 st.header("**Relatedness per Ancestry**")
#                 st.plotly_chart(bar_3)
#             st.markdown('---')
            

#             left_col2, right_col2 = st.columns([1.25,1])


#             st.header('QC Step 3: Variant Filtering')
#             var_exp = st.expander("Description", expanded=False)
#             with var_exp:
#                 st.write('\
#                         1. Genotype Missingness <= 0.05  \n\
#                         2. Case/Control Missingness: P > 1e-4  \n\
#                         3. Haplotype Missingness: P > 1e-4  \n\
#                         4. Hardy-Weinberg Equilibrium: P > 1e-4  \n\
#                 ')

#             st.header("**Variant Filtering per Ancestry**")
#             st.plotly_chart(bar_6)
#             st.markdown('---')


#             if df_6.shape[0]>0:
#                 st.markdown("**Failed Prune Steps**")
#                 st.table(df_6)
#                 st.markdown('---')
            