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

st.markdown('### **Reference Panel Admixture Populations**')

ref_admix = pd.read_csv('data/ref_panel_admixture.txt', sep='\s+')

st.dataframe(ref_admix)


# f, ax = plt.subplots(1, 1, figsize=(14, 2), facecolor="w", constrained_layout=True, dpi=300)
# admix_plot = admixtureplot(data=load_dataset("admixture_output.Q"), 
#               population_info=load_dataset("admixture_population.info"),
#               ylabel_kws={"rotation": 45, "ha": "right"},
#               ax=ax)
admix_pop_info = ref_admix['ancestry']
admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10']]
# admixture_output.columns = pd.RangeIndex(admixture_output.columns.size)

st.image('data/refpanel_admix.png', caption='Reference Panel Admixture Plot')
