import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from hold_data import place_logos, config_page


config_page('Home')

# Place logos in sidebar
place_logos()

# Main title
st.markdown("<h1 style='text-align: center; color: #0f557a; font-family: Helvetica; '>GP2 Internal Cohort Browser</h1>", unsafe_allow_html=True)

# Page formatting 
sent1, sent2, sent3 = st.columns([1,6,1])  # holds brief overview sentences
exp1, exp2, exp3 = st.columns([1, 2, 1])  # holds expander for full description

# sent2.markdown("##### Interactive tool to visualize quality control and ancestry prediction summary statistics across all GP2 cohorts. #####")
sent2.markdown("<h5 style='text-align: center; '>Interactive tool to visualize quality control and ancestry prediction summary statistics\
             across all GP2 cohorts. Please select a page marked with \U0001F9EC in the sidebar to begin.</h1>", unsafe_allow_html=True)

# Display expander with full project description
overview = exp2.expander("Full Description", expanded=False)
with overview:
    st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
    
    st.markdown("## _GenoTools Preprint Out Now!_")
    st.markdown('For a more in-depth description of the methods used to process the GP2 releases, please see the GenoTools preprint: \
                 https://www.biorxiv.org/content/10.1101/2024.03.26.586362v1')

    st.markdown("## _Quality Control_")
    st.markdown('### _Sample-Level Pruning_')
    st.markdown('Genotypes are pruned for call rate with maximum sample genotype missingness of 0.02 (--mind 0.02). Samples which pass\
                 call rate pruning are then pruned for discordant sex where samples with 0.25 <= sex F <= 0.75 are pruned. Sex F < 0.25\
                 are female and Sex F > 0.75 are male. Samples that pass sex pruning are then differentiated by ancestry (refer to\
                 ancestry method below). Per-ancestry genotypes are then pruned for genetic relatedness using KING, where a  cutoff of 0.0884 \
                 was used to determine second degree relatedness and 0.354 is used to determine duplicates. For purposes of imputation,\
                 related samples are left in and duplicated samples are pruned. Next, samples are pruned for heterozygosity where F <= -0.25 of\
                 F>= 0.25.')
    st.markdown('### _Variant-Level Pruning_')
    st.markdown('Variants are pruned for missingness by case-control where P<=1e-4 to detect platform/batch differences in case-control status.\
                 Next, variants are pruned for missingness by haplotype for flanking variants where P<=1e-4. Lastly, controls are filtered for HWE\
                 at a threshold of 1e-4.')
    st.markdown("## _Ancestry_")
    st.markdown('### _Reference Panel_')
    st.markdown('The reference panel is composed of 4008 samples from 1000 Genomes Project, Human Genome Diversity Project (HGDP), \
                 and an Ashkenazi Jewish reference panel (Gene Expression Omnibus (GEO) database, www.ncbi.nlm.nih.gov/geo \
                (accession no. GSE23636)) (REF) with the following ancestral makeup:')
    st.markdown(
                """
                - African (AFR): 819
                - African Admixed and Caribbean (AAC): 74
                - Ashkenazi Jewish (AJ): 471
                - Central Asian (CAS): 183
                - East Asian (EAS): 585
                - European (EUR): 534
                - Finnish (FIN): 99
                - Latino/American Admixed (AMR): 490
                - Middle Eastern (MDE): 152
                - South Asian (SAS): 601
                """
                )
    st.markdown('Samples were chosen from 1000 Genomes and HGDP to match the specific ancestries present in GP2. The reference panel was then\
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
    st.markdown('A classifier was then trained using UMAP transformations of the PCs and a linear XGBoost classifier using a 5-fold\
                cross-validation using an sklearn pipeline and scored for balanced accuracy with a gridsearch over the following parameters:')
    st.markdown(
                """
                - “umap__n_neighbors”: [5,20]
                - “umap__n_components”: [15,25]
                - “umap__a”: [0.75, 1.0, 1.5]
                - “umap__b”: [0.25, 0.5, 0.75]
                - “xgboost__lambda”: [0.001, 0.01, 0.1, 1, 10, 100]
                """
                )
    st.markdown('Performance varies from 95-98% balanced accuracy on the test set depending on overlapping genotypes.')

    st.markdown('### _Prediction_')
    st.markdown('Scaled PCs for genotypes are transformed using UMAP trained fitted by the training set and then predicted by the classifier. \
                 Genotypes are split and output into individual ancestries. Prior to release 5, AAC and AFR labels were combined into a \
                 single category and ADMIXTURE 1 (v1.3.0-https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz) \
                was run using the --supervised functionality to further divide these two categories where AFR was assigned if AFR admixture \
                was >=90% and AAC was assigned if AFR admixture was <90%. From release 5 on, the AFR and AAC sample labels in the reference panel \
                 are adjusted using a perceptron model, and the predictions based on the updated reference panel labels effectively estimate the \
                results from the ADMIXTURE step that was previously used.')
    
    st.markdown('### _Complex Admixture History_')
    st.markdown('Certain highly admixed ancestry groups are not well-represented by the constructed reference panel used by GenoTools. Due a lack of publicly available \
                 reference samples for highly admixed groups, GenoTools employs a method to identify samples of this nature and place them in an ancestry  \
                 group that is not present in the reference panel, named “Complex Admixture History” (CAH). Highly admixed samples of this nature should be analyzed \
                 independently. Since there are no reference samples to base the prediction of CAH ancestry on, a PC-based approach is used instead. Using the training \
                 data, the PC centroid of each reference panel ancestry group is calculated, along with the overall PC centroid. For each new sample, the PC distance from \
                 each centroid is then calculated. Any sample whose PC distance is closer to the overall PC centroid of the training data than to any reference panel \
                 ancestry group centroid is labeled as CAH.')

    st.caption('**_References_**')
    st.caption('_D.H. Alexander, J. Novembre, and K. Lange. Fast model-based estimation of ancestry in unrelated individuals. Genome Research, 19:1655–1664, 2009._')

# Customize text in Expander element
hvar = """ <script>
                var elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
                elements[0].style.fontSize = 'large';
                elements[0].style.color = '#0f557a';
            </script>"""
components.html(hvar, height=0, width=0)

