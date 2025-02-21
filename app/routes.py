import os
from fastapi import APIRouter, Query, Depends
from app.embeddings.embed_model import search_similar  # Relative import
from google.cloud import aiplatform  # Not used in this version, but keep it if needed later
from sentence_transformers import SentenceTransformer
import requests
import faiss

router = APIRouter()

def get_model():
    from app.main import model  # Relative import
    return model

def get_index():
    from app.main import index  # Relative import
    return index

def get_metadata():
    from app.main import metadata  # Relative import
    return metadata

@router.get("/ask")
async def ask_question(
    question: str,
    books: list[str] = Query(["Bhagavad Gita", "Quran", "Bible"]),
    model: SentenceTransformer = Depends(get_model),
    index: faiss.IndexFlatL2 = Depends(get_index),
    metadata: list = Depends(get_metadata)
):
    """Endpoint to answer a question."""

    local_results = search_similar(question, model, index, metadata, books)

    gemini_results = await call_gemini_api(question, local_results)
    return {"local_results": local_results, "gemini_results": gemini_results}

async def call_gemini_api(question: str, local_results):
    """Calls the Gemini AI API using the API key."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY environment variable not set."}

    model_name = os.environ.get("MODEL_NAME")
    api_url = f"https://asia-south1-aiplatform.googleapis.com/v1/projects/{os.environ.get('PROJECT_ID')}/locations/{os.environ.get('LOCATION')}/endpoints/{model_name}:predict"

    context_str = "\n\n".join([f"{result['book']} - {result['verse']}: {result['meaning']}" for result in local_results]) if local_results else "No relevant context found."

    prompt = f"""Answer the question based on the provided context from the holy books:

    Question: {question}

    Context:
    {context_str}

    Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "instances": [{"prompt": prompt}],
        "parameters": {}  # Add any parameters if needed
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        gemini_response = response.json()
        return gemini_response["predictions"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"Failed to call Gemini API: {e}"}
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}, Response: {response.text}")
        return {"error": f"Failed to parse Gemini response: {e}"}

