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

st.set_page_config(page_title = "Admixture Populations", layout = 'wide')

frontend_bucket_name = 'frontend_app_materials'
frontend_bucket = get_gcloud_bucket(frontend_bucket_name)

st.markdown('## **Reference Panel Admixture Populations**')

ref_admix = blob_as_csv(frontend_bucket, 'ref_panel_admixture.txt')

st.dataframe(ref_admix)

admix_pop_info = ref_admix['ancestry']
admixture_output = ref_admix[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10']]

admix_plot = frontend_bucket.get_blob('refpanel_admix.png')
admix_plot = admix_plot.download_as_bytes()
st.image(admix_plot)
