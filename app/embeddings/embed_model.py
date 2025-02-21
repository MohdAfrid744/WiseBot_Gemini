import numpy as np
import faiss

def generate_embeddings(datasets, model):  # model is now passed as an argument
    """Generate embeddings for all verses."""
    embeddings = []
    metadata = []

    for book, verses_list in datasets.items():
        for entry in verses_list:
            embedding = model.encode(entry["verse"])
            embeddings.append(embedding)
            metadata.append({
                "book": book,
                "verse": entry["verse"],
                "meaning": entry.get("meaning"),  # Handle missing "meaning"
                "chapter": entry.get("chapter")  # Include chapter if available.
            })

    return np.array(embeddings), metadata

def create_faiss_index(embeddings):
    """Create and return a FAISS index."""
    if embeddings.size == 0:  # Check if the embeddings array is empty.
        return None  # Return None if there are no embeddings.

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # consider using IndexIVFFlat for large datasets.
    index.add(embeddings)
    return index

def search_similar(query, model, index, metadata, selected_books, top_k=3):  # Accept the index
    """Search for similar verses using FAISS."""
    if index is None:  # Handle case when there are no embeddings.
        return []

    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:  # indices is a 2D array
        entry = metadata[idx]
        if entry["book"] in selected_books:
            results.append(entry)

    return results

