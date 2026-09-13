import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="Tag Finder AI", page_icon="🔍", layout="centered")

# LaBSE is specialized for Banglish, Bangla, and English semantic search
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/LaBSE')

# Read CSV safely with fallback encoding
def read_csv_safely(file_path):
    for enc in ['utf-8-sig', 'utf-8', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(file_path, encoding='iso-8859-1')

# Load and Encode Tag Data
@st.cache_data
def load_data(_model):
    df = read_csv_safely('tags.csv')
    
    tag_col = df.columns[0]
    eng_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    bn_col = df.columns[2] if len(df.columns) > 2 else eng_col
    
    # Handle missing/empty values
    df[tag_col] = df[tag_col].fillna('').astype(str)
    df[eng_col] = df[eng_col].fillna('').astype(str)
    df[bn_col] = df[bn_col].fillna('').astype(str)
    
    # Combined text for AI embedding
    df['combined_text'] = df[tag_col] + " " + df[eng_col] + " " + df[bn_col]
    
    embeddings = _model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return df, embeddings, tag_col, eng_col, bn_col

st.title("🔍 AI-Powered Tag Matcher")
st.write("Bangla, Banglish ba English-e search kore relevant tag khunje nin.")

with st.spinner("Loading AI Model (LaBSE)..."):
    model = load_model()
    df, tag_embeddings, tag_col, eng_col, bn_col = load_data(model)

# Search Input
query = st.text_input("Search:", placeholder="e.g., user er bike lagbe tai cancel korte cacche")

top_n = st.slider("Max results to show:", min_value=1, max_value=5, value=3)
similarity_threshold = st.slider("Minimum Match Threshold (%)", min_value=30, max_value=80, value=45)

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
    
    # Sort results
    top_results = torch.topk(cosine_scores, k=min(top_n, len(df)))
    
    st.subheader("Match Results:")
    found = False
    
    for score, idx in zip(top_results.values, top_results.indices):
        score_val = float(score) * 100
        
        # Filter out low match / irrelevent tags
        if score_val >= similarity_threshold:
            found = True
            row = df.iloc[int(idx)]
            
            with st.container():
                st.markdown(f"### 🏷️ **{row[tag_col]}** `(Match: {score_val:.1f}%)`")
                st.write(f"**English:** {row[eng_col]}")
                st.write(f"**বাংলা:** {row[bn_col]}")
                st.divider()
                
    if not found:
        st.warning(f"No match found above {similarity_threshold}% relevance.")