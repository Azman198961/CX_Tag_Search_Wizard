import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Tag Finder AI", page_icon="🔍", layout="centered")

# Multilingual AI Model
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

# Comprehensive Banglish to CX Context Vocabulary Mapper
BANGLISH_CX_MAP = {
    # Money & Payment related
    "taka": "money paid fare price টাকা অর্থ",
    "take": "money paid fare price টাকা",
    "nise": "paid collected took নিয়াছে নিয়েছে নেওয়া",
    "niche": "paid collected took নিয়াছে নিয়েছে",
    "neyeche": "paid collected took নিয়াছে নিয়েছে",
    "caise": "asked demanded charged চাইছে চেয়েছে চাওয়া",
    "chaise": "asked demanded charged চাইছে চেয়েছে",
    "chay": "asked demanded চায়",
    "chaye": "asked demanded চায়",
    "dibena": "not paid refuse দিবে না দেয়নি",
    "deni": "not paid refuse দেয়নি",
    "dini": "not paid refuse দেয়নি",
    "dilo": "paid gave দিল দিয়েছে",
    "paise": "received got পেয়েছে পাইছে",
    "extra": "extra excess overcharged অতিরিক্ত বেশি",
    "beshi": "extra excess overcharged অতিরিক্ত বেশি",
    "bashi": "extra excess overcharged অতিরিক্ত বেশি",
    "overcharge": "extra money customer paid excessive",

    # Ride, Trip & Delivery related
    "rider": "rider driver captain ড্রাইভার রাইডার",
    "gard": "rider driver guard রাইডার",
    "driver": "rider driver ড্রাইভার",
    "gari": "vehicle ride car bike গাড়ি বাইক",
    "bike": "bike motorcycle বাইক",
    "trip": "ride trip ভ্রমণ ট্রিপ",
    "ride": "ride trip রাইড ট্রিপ",

    # Status & Issues
    "cancel": "cancellation cancelled বাতিল ক্যানসেল",
    "cancle": "cancellation cancelled বাতিল",
    "late": "delay 4 hours slow সময় বেশি দেরি",
    "deri": "delay late 4 hours দেরি সময়",
    "somoy": "time delay hours সময়",
    "location": "address destination লোকেশন ঠিকানা",
    "behavior": "unprofessional rude misbehavior খারাপ ব্যবহার আচরণ",
    "kharaap": "bad rude misbehavior খারাপ",
    "kharap": "bad rude misbehavior খারাপ",
    "app": "application system app প্রযুক্তি অ্যাপ",
    "problem": "issue error problem সমস্যা",
    "somossa": "issue error problem সমস্যা",
    "bhara": "fare money price ভাড়া"
}

def normalize_and_expand_query(user_query, score_cutoff=70.0):
    """
    Translates Banglish tokens dynamically using RapidFuzz for fuzzy matching
    to handle typos and extra letters (e.g., 'extrra', 'doubble') before feeding into AI.
    """
    words = re.findall(r'\w+', user_query.lower())
    expanded_terms = [user_query] # Keep original text
    dict_keys = list(BANGLISH_CX_MAP.keys())
    
    for word in words:
        if word in BANGLISH_CX_MAP:
            expanded_terms.append(BANGLISH_CX_MAP[word])
        else:
            # RapidFuzz match against Banglish dictionary keys
            match = process.extractOne(word, dict_keys, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
            if match:
                best_matched_key = match[0]
                expanded_terms.append(BANGLISH_CX_MAP[best_matched_key])
                
    return " ".join(expanded_terms)

# Load and process data
@st.cache_data(ttl=300)
def load_data(_model):
    df = read_csv_safely('tags.csv')
    df = df.fillna('').astype(str)
    
    tag_col = df.columns[0]
    eng_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    bn_col = df.columns[2] if len(df.columns) > 2 else eng_col
    
    # Combined vector text for embedding
    df['combined_text'] = df[tag_col] + " | " + df[eng_col] + " | " + df[bn_col]
    
    embeddings = _model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return df, embeddings, tag_col, eng_col, bn_col

st.title("🔍 AI-Powered Tag Matcher")
st.write("Bangla, English ba Banglish-e search kore relevant tag khunje nin.")

with st.spinner("Loading AI Engine..."):
    model = load_model()
    df, tag_embeddings, tag_col, eng_col, bn_col = load_data(model)

# Search Input
query = st.text_input("Search:", placeholder="e.g., extrra taka nise, rider extra taka caise, 4 hours delay")

top_n = st.slider("Max results to show:", min_value=1, max_value=10, value=5)
similarity_threshold = st.slider("Minimum Match Threshold (%)", min_value=5, max_value=80, value=15)

if query:
    # Transform Banglish/Messy text using RapidFuzz dictionary expansion
    enriched_search_query = normalize_and_expand_query(query)
    
    # AI Embedding Score
    query_embedding = model.encode(enriched_search_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
    final_scores = cosine_scores.clone()
    
    # Hybrid Score Boost using RapidFuzz directly on full text records
    query_words = set(re.findall(r'\w+', query.lower()))
    
    for idx, text in enumerate(df['combined_text']):
        text_lower = text.lower()
        
        # 1. Exact Word overlap boost
        matched_count = sum(1 for word in query_words if word in text_lower)
        if query_words:
            final_scores[idx] += (matched_count / len(query_words)) * 0.15
            
        # 2. RapidFuzz partial character overlap score boost (handles typos in csv text itself)
        fuzzy_score = fuzz.partial_ratio(query.lower(), text_lower) / 100.0
        final_scores[idx] += fuzzy_score * 0.15

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
