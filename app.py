import os
import pickle

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
TOP_K = 6

# Set to the task type you used when you created vectors.npy.
# If you embedded the documents with no task type, set this to None.
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

SYSTEM_PROMPT = """You are an advisor who helps a team make decisions in a business simulation game.
Answer the user's question using ONLY the context excerpts provided.

Style:
- Give a clear, direct recommendation, highlighting the pros and cons.
- Explain your reasoning using the specific numbers and rules from the context.
- Say what matters more and what matters less, and suggest a next step.
- Keep it short (6-12 sentences) and conversational.
- If the context does not contain enough information, say so plainly instead of guessing.
- Do not mention "context" or "excerpts"; speak as if you know the material."""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@st.cache_resource
def load_index():
    vectors = np.load("vectors.npy").astype("float32")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine on unit vectors
    index.add(vectors)
    return index, chunks


def embed_query(text):
    config = types.EmbedContentConfig(task_type=QUERY_TASK_TYPE) if QUERY_TASK_TYPE else None
    res = client.models.embed_content(model=EMBED_MODEL, contents=text, config=config)
    vec = np.array([res.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


index, chunks = load_index()

st.title("📚 Facts Repository Search")
query = st.text_area("Ask a question or describe your situation:", height=100)

if query:
    with st.spinner("Thinking..."):
        scores, ids = index.search(embed_query(query), TOP_K)
        matches = [chunks[i] for i in ids[0]]
        context = "\n\n---\n\n".join(
            f"[{m['file']} | chunk {m['chunk']}]\n{m['text']}" for m in matches
        )
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"Context:\n{context}\n\nQuestion:\n{query}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.3
            ),
        )

    st.markdown(response.text)

    with st.expander("Sources used"):
        for score, m in zip(scores[0], matches):
            st.markdown(f"**{m['file']}** | chunk {m['chunk']} | similarity {score:.3f}")
            st.write(m["text"])
            st.divider()
