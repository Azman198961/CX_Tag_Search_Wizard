import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="Tag Finder AI", page_icon="🔍", layout="centered")

# Multilingual AI Model Load
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Safe CSV reader with encoding fallback
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
    
    # Explicit Column Names based on your CSV
    tag_col = 'Tag' if 'Tag' in df.columns else df.columns[0]
    eng_col = 'English description' if 'English description' in df.columns else df.columns[1]
    bn_col = 'বাংলা description' if 'বাংলা description' in df.columns else (df.columns[2] if len(df.columns) > 2 else eng_col)
    
    # Combined text representation for vector generation
    df['combined_text'] = df[tag_col] + " | " + df[eng_col] + " | " + df[bn_col]
    
    embeddings = _model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return df, embeddings, tag_col, eng_col, bn_col

st.title("🔍 AI-Powered Tag Matcher")
st.write("Bangla, English ba Banglish-e search kore relevant tag khunje nin.")

with st.spinner("Loading AI Model..."):
    model = load_model()
    df, tag_embeddings, tag_col, eng_col, bn_col = load_data(model)

# Search Input
query = st.text_input("Search:", placeholder="e.g., 4 hours, cancellation fee, Double Payment")

top_n = st.slider("Max results to show:", min_value=1, max_value=10, value=5)
similarity_threshold = st.slider("Minimum Match Threshold (%)", min_value=10, max_value=80, value=25)

if query:
    query_clean = query.strip().lower()
    
    # AI Similarity Matching
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
    
    # Hybrid Boost: Text similarity-র পাশাপাশি Exact Keyword থাকলে Match Score বাড়িয়ে দেওয়া
    final_scores = cosine_scores.clone()
    for idx, text in enumerate(df['combined_text']):
        if query_clean in text.lower():
            final_scores[idx] += 0.40  # Boost score by 40% if keyword matches exactly
            
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
        st.warning(f"No match found above {similarity_threshold}% relevance. Lower the threshold slider to see more.")
