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

from hold_data import blob_as_csv, get_gcloud_bucket, cohort_select


if 'gp2_removebg' in st.session_state:
    st.set_page_config(
        page_title="Quality Control",
        page_icon=st.session_state.gp2_bg,
        layout="wide",
    )
else: 
    st.set_page_config(
        page_title="Quality Control",
        layout="wide"
    )