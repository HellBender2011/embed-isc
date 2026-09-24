import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
import os

# Configure Gemini
from google.colab import userdata
genai.configure(api_key=userdata.get('geminikey'))
model = "models/gemini-embedding-001"

folder = "/content/drive/MyDrive/repo"

def load_and_chunk(folder, chunk_size=900):
    chunks = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                text = f.read()
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append({"file": fname, "chunk": i//chunk_size, "text": chunk})
    return chunks

chunks = load_and_chunk(folder)

embeddings = []
for c in chunks:
    e = genai.embed_content(model=model, content=c["text"])
    embeddings.append(e['embedding'])

vectors = np.array(embeddings).astype("float32")
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
