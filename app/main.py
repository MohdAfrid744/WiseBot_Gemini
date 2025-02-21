import os
from fastapi import FastAPI
from app.routes import router
from app.dataset_loader import load_all_datasets
from app.embeddings.embed_model import generate_embeddings, create_faiss_index
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

print(f"CWD in main.py: {os.getcwd()}")

load_dotenv()  # Load environment variables from .env

app = FastAPI(
    title="Holy Chatbot API",
    description="An API that provides answers from Bhagavad Gita, Quran, and Bible.",
    version="1.0.0"
)

# Initialize global variables
model = None
index = None
metadata: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    global model, index, metadata

    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        exit()

    DATASETS = load_all_datasets()
    if DATASETS is None:
        print("Failed to load datasets. Exiting.")
        exit()

    embeddings, metadata = generate_embeddings(DATASETS, model)
    index = create_faiss_index(embeddings)

    if index is None:
        print("Failed to create FAISS index. Exiting.")
        exit()

    print("Application started.")

app.include_router(router)

