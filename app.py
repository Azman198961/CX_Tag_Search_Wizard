import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re

st.set_page_config(page_title="Tag Finder AI", page_icon="🔍", layout="centered")

# Multilingual AI Model (Best for Bangla, Banglish & English meaning matching)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Safe CSV reader
def read_csv_safely(file_path):
    for enc in ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(file_path)

# Load and process data
@st.cache_data(ttl=300)
def load_data(_model):
    df = read_csv_safely('tags.csv')
    df = df.fillna('').astype(str)
    
    # Standardize column names dynamically
    cols = df.columns.tolist()
    tag_col = cols[0]
    eng_col = cols[1] if len(cols) > 1 else cols[0]
    bn_col = cols[2] if len(cols) > 2 else eng_col
    extra_col = cols[3] if len(cols) > 3 else None  # Optional Banglish/Search keywords column
    
    # Build enriched context for AI vector embedding
    if extra_col:
        df['combined_text'] = df[tag_col] + " | " + df[eng_col] + " | " + df[bn_col] + " | " + df[extra_col]
    else:
        df['combined_text'] = df[tag_col] + " | " + df[eng_col] + " | " + df[bn_col]
    
    embeddings = _model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return df, embeddings, tag_col, eng_col, bn_col

st.title("🔍 AI-Powered Tag Matcher")
st.write("Bangla, English ba Banglish-e search kore relevant tag khunje nin.")

with st.spinner("Loading AI Model..."):
    model = load_model()
    df, tag_embeddings, tag_col, eng_col, bn_col = load_data(model)

# Search Input
query = st.text_input("Search:", placeholder="Type anything in English, Bangla, or Banglish...")

top_n = st.slider("Max results to show:", min_value=1, max_value=10, value=5)
similarity_threshold = st.slider("Minimum Match Threshold (%)", min_value=10, max_value=80, value=20)

if query:
    query_clean = query.strip().lower()
    query_tokens = [w for w in re.findall(r'\w+', query_clean) if len(w) > 1]
    
    # 1. AI Vector Similarity Search
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
    final_scores = cosine_scores.clone()
    
    # 2. Dynamic Word-overlap & Character-ngram Fuzzy Match
    # Fits any tag without hardcoded dictionary
    for idx, text in enumerate(df['combined_text']):
        text_lower = text.lower()
        
        # Word token overlap boost
        matched_tokens = sum(1 for token in query_tokens if token in text_lower)
        if query_tokens:
            token_ratio = matched_tokens / len(query_tokens)
            final_scores[idx] += (token_ratio * 0.30)
            
        # Substring/exact phrase boost
        if query_clean in text_lower:
            final_scores[idx] += 0.25

    top_results = torch.topk(final_scores, k=min(top_n, len(df)))
    
    st.subheader("Match Results:")
    found = False
    
    for score, idx in zip(top_results.values, top_results.indices):
        score_val = min(float(score) * 100, 100.0)
        
        if score_val >= similarity_threshold:
            found = True
            row = df.iloc[int(idx)]
            
            with st.container():
                st.markdown(f"### 🏷️ **{row[tag_col]}** `(Match: {score_val:.1f}%)`")
                st.write(f"**English:** {row[eng_col]}")
                st.write(f"**বাংলা:** {row[bn_col]}")
                st.divider()
                
    if not found:
        st.warning(f"No match found above {similarity_threshold}% relevance. Try lowering the threshold slider.")
