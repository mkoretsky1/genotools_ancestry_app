import os
import sys
import subprocess
# from git import blob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from PIL import Image
import datetime
from io import StringIO

from google.cloud import storage

from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'genotools-02f64a1e10be.json'

def blob_as_csv(bucket, path, header='infer'):
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep='\s+', header=header)
    return df

def get_gcloud_bucket(bucket_name):
    storage_client = storage.Client(project='genotools')
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

bucket_name = 'frontend_app_data'
bucket = get_gcloud_bucket(bucket_name)