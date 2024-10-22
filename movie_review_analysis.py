import os
import pandas as pd
import torch
import transformers


df_review = pd.read_csv("IMDB_Dataset.csv")
df_review.head(3)
df_review.iloc[0].review

transformers.utils.logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import AutoModelForCausalLM, AutoTokenizer


phi3_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",torch_dtype="auto", trust_remote_code=True)