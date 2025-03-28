import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIG ===
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "chunk_vectors.npy"
INDEX_FILE = "faiss.index"
CHUNK_MAP_FILE = "chunk_map.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o"

# Initialize OpenAI client
client = OpenAI(api_key="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")

# === STEP 1: Load textbook chunks ===
try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
except FileNotFoundError:
    print(f"Error: {CHUNKS_FILE} not found. Please run chunking.py first.")
    exit(1)

# === STEP 2: Load or compute embeddings & index ===
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
    print("[1/4] Loading cached embeddings and index...")
    chunk_vectors = np.load(EMBEDDINGS_FILE)
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNK_MAP_FILE, "rb") as f:
        chunk_map = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
else:
    print("[1/4] Embedding textbook chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_vectors = model.encode(chunks, convert_to_numpy=True)

    print("[2/4] Creating FAISS index...")
    dimension = chunk_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_vectors)

    print("[3/4] Saving index and embeddings to disk...")
    np.save(EMBEDDINGS_FILE, chunk_vectors)
    faiss.write_index(index, INDEX_FILE)
    chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
    with open(CHUNK_MAP_FILE, "wb") as f:
        pickle.dump(chunk_map, f)

# === STEP 3: Define search and LLM answer ===
def search_chunks(query, top_k=6):
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [chunk_map[i] for i in I[0]]

def ask_accounting_ai(question):
    # Get relevant chunks from vector store
    docs = index.similarity_search(question, k=3)
    
    # Extract content from chunks and join
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format prompt
    prompt = f"""You are a Belgian accounting assistant specializing in co-ownership accounting. 
Answer the following question based on the Belgian co-ownership accounting plan.
Use only the information from the provided context. If you cannot answer from the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# === STEP 4: CLI interface ===
if __name__ == "__main__":
    print("🇧🇪 Accounting Assistant AI — Based on the Belgian Co-Ownership Accounting Plan\n")
    while True:
        try:
            question = input("❓ Your question (or type 'exit'): ")
            if question.lower().strip() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            answer = ask_accounting_ai(question)
            print("\n🤖 Answer:")
            print(answer)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
