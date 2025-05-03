# resources.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Global variables to store models and data
faiss_index = None
embedding_model = None
med_mcqa_df = None