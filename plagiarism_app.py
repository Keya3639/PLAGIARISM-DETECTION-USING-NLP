import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


@st.cache_resource(show_spinner="Loading Sentence Transformer Model...")
def load_model():
    """Load the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Loading data, generating embeddings, and building FAISS index...")
def load_data_and_build_index():
    """Load the data, combine text columns, generate embeddings, and build the FAISS index."""
    
    file_name = 'www.csv'
    
    try:
        df = pd.read_csv(file_name)
        if 'text1' not in df.columns or 'text2' not in df.columns:
            st.error(f"FATAL ERROR: Expected columns 'text1' and 'text2' not found.")
            st.warning(f"Available columns are: {list(df.columns)}.")
            return None, None, None
        texts_to_index_1 = df['text1'].astype(str).tolist()
        texts_to_index_2 = df['text2'].astype(str).tolist()
        all_source_texts = list(set(texts_to_index_1 + texts_to_index_2))      
        st.info(f"Database built from {len(texts_to_index_1) + len(texts_to_index_2)} sentences (Unique: {len(all_source_texts)})")
        

        model = load_model()
        embeddings = model.encode(all_source_texts, show_progress_bar=False)
        index_data = np.array([e for e in embeddings]).astype('float32')
        embedding_dim = embeddings.shape[1] 
        index = faiss.IndexFlatL2(embedding_dim) 
        index.add(index_data)
        return model, index, all_source_texts
    
    except FileNotFoundError:
        st.error(f"Error: '{file_name}' not found. Please place it in the same directory as plagiarism_app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        return None, None, None

def check_plagiarism(query_text, model, index, texts_to_index, k, threshold):
    """Function to perform the FAISS search and report findings."""
    
    if not query_text:
        return []
    query_embedding = model.encode([query_text]).astype('float32')
    D, I = index.search(query_embedding, k)
    
    results = []
    
    for rank in range(k):
        distance = D[0][rank]
        index_in_dataset = I[0][rank]
        retrieved_text = texts_to_index[index_in_dataset]
        is_plagiarized = distance < threshold
        
        results.append({
            'Similarity (L2 Distance)': f"{distance:.4f}",
            'Source Text': retrieved_text,
            'Is Plagiarized': is_plagiarized
        })
        
    return results

st.set_page_config(page_title="Semantic Plagiarism Detector", layout="wide")
st.title("Plagiarism Detection Tool using semantic similarity")
st.markdown("Detecting copied or heavily paraphrased text using **Sentence Embeddings** and **FAISS**.")

model, index, texts_to_index = load_data_and_build_index()

if model is not None and index is not None:

    st.subheader("Text to Check for Plagiarism")
    query_text = st.text_area("Paste the text you want to check here:", height=150)
    st.sidebar.header("Detection Settings")
    k_neighbors = st.sidebar.slider("Number of Nearest Neighbors (k)", min_value=1, max_value=10, value=3)
    similarity_threshold = st.sidebar.slider("Plagiarism Threshold (L2 Distance)", 
                                             min_value=0.01, max_value=1.0, value=0.3, step=0.01)
    st.sidebar.info(f"Lower L2 Distance = Higher Similarity. Texts with a distance below **{similarity_threshold}** are flagged.")

    if st.button("Run Plagiarism Check"):
        if query_text:
            with st.spinner('Encoding text and searching database...'):
                results = check_plagiarism(query_text, model, index, texts_to_index, k_neighbors, similarity_threshold)

            st.subheader("Results (Top Nearest Neighbors)")

            for res in results:
                col1, col2 = st.columns([1, 4])
                
                status = "**PLAGIARISM FLAG**" if res['Is Plagiarized'] else "**ORIGINAL/LOW SIMILARITY**"
                col1.markdown(status)
                
                col2.markdown(f"**L2 Distance:** `{res['Similarity (L2 Distance)']}` (Threshold: < {similarity_threshold})")
                col2.markdown(f"**Source Text Found:** *{res['Source Text']}*")
                st.markdown("---")
            
            if any(r['Is Plagiarized'] for r in results):
                st.error("One or more texts were flagged as highly similar to the source database.")
            else:
                st.success("No highly similar texts found below the set threshold.")

        else:
            st.warning("Please enter text to check.")