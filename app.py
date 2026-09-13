import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from thefuzz import fuzz

st.set_page_config(page_title="Tag Finder AI", page_icon="🔍", layout="centered")

# Multilingual AI Model Load
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

# Dictionary for Banglish & Common Misspellings to English mapping
BANGLISH_DICTIONARY = {
    "taka": "money টাকা",
    "take": "money টাকা",
    "nise": "paid took নিয়াছে নিয়েছে",
    "niche": "paid took নিয়াছে নিয়েছে",
    "neyeche": "paid took নিয়াছে নিয়েছে",
    "caise": "asked demanded চাইছে চেয়েছে",
    "chaise": "asked demanded চাইছে চেয়েছে",
    "chay": "asked demanded চায়",
    "dibena": "did not pay দেয়নি দিবে না",
    "deni": "did not pay দেয়নি",
    "extra": "extra অতিরিক্ত",
    "beshi": "extra অতিরিক্ত বেশি",
    "bashi": "extra অতিরিক্ত বেশি",
    "cancel": "cancellation ক্যানসেল বাতিল",
    "late": "delay 4 hours সময় বেশি",
    "somoy": "time hours সময়",
    "gard": "rider driver রাইডার",
    "driver": "rider ড্রাইভার রাইডার",
    "app": "drive app অ্যাপ",
    "bhara": "fare money ভাড়া"
}

def translate_banglish(query):
    words = query.lower().split()
    translated_words = []
    for word in words:
        # Exact or close fuzzy match with banglish dictionary
        matched = False
        for key in BANGLISH_DICTIONARY:
            if word == key or fuzz.ratio(word, key) > 85:
                translated_words.append(BANGLISH_DICTIONARY[key])
                matched = True
                break
        if not matched:
            translated_words.append(word)
    return " ".join(translated_words)

# Load and process data
@st.cache_data(ttl=300)
def load_data(_model):
    df = read_csv_safely('tags.csv')
    df = df.fillna('').astype(str)
    
    tag_col = 'Tag' if 'Tag' in df.columns else df.columns[0]
    eng_col = 'English description' if 'English description' in df.columns else df.columns[1]
    bn_col = 'বাংলা description' if 'বাংলা description' in df.columns else (df.columns[2] if len(df.columns) > 2 else eng_col)
    
    df['combined_text'] = df[tag_col] + " | " + df[eng_col] + " | " + df[bn_col]
    
    embeddings = _model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return df, embeddings, tag_col, eng_col, bn_col

st.title("🔍 AI-Powered Tag Matcher")
st.write("Bangla, English ba Banglish-e search kore relevant tag khunje nin.")

with st.spinner("Loading AI Model..."):
    model = load_model()
    df, tag_embeddings, tag_col, eng_col, bn_col = load_data(model)

# Search Input
query = st.text_input("Search:", placeholder="e.g., extra taka nise, extra taka caise, cancellation fee")

top_n = st.slider("Max results to show:", min_value=1, max_value=10, value=5)
similarity_threshold = st.slider("Minimum Match Threshold (%)", min_value=10, max_value=80, value=25)

if query:
    # Convert Banglish query into enriched search context
    enriched_query = translate_banglish(query)
    
    # AI Vector Similarity Matching
    query_embedding = model.encode(enriched_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
    
    final_scores = cosine_scores.clone()
    
    # Fuzzy & Keyword Matching Boost Logic
    query_tokens = re.findall(r'\w+', query.lower())
    
    for idx, text in enumerate(df['combined_text']):
        text_lower = text.lower()
        
        # 1. Exact query match boost
        if query.lower() in text_lower:
            final_scores[idx] += 0.35
            
        # 2. Token/Word level match boost (e.g., matching both 'extra' and 'money')
        matched_tokens = sum(1 for token in query_tokens if token in text_lower or fuzz.partial_ratio(token, text_lower) > 80)
        if len(query_tokens) > 0:
            token_match_ratio = matched_tokens / len(query_tokens)
            final_scores[idx] += (token_match_ratio * 0.25)

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
