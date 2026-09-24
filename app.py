import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
import os

# Configure Gemini
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = "models/gemini-embedding-001"
# Load precomputed vectors and chunks
vectors = np.load("vectors.npy")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Build FAISS index
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

st.title("📚 Facts Repository Search")

query = st.text_input("Ask a question:")

if query:
    q_embed = genai.embed_content(model=model, content=query)['embedding']
    D, I = index.search(np.array([q_embed]).astype("float32"), k=5)

    st.write("### Top Matches")
    for idx in I[0]:
        match = chunks[idx]
        st.write(f"**File:** {match['file']} | **Chunk:** {match['chunk']}")
        st.write(match['text'][:500] + "...")