import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import time


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))


def euclidean_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes similarity based on Euclidean distance (higher is more similar)."""
    distance = euclidean_distance(vector_a, vector_b)
    # Convert distance to similarity (0-1 scale, higher is more similar)
    return 1 / (1 + distance)


def manhattan_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes similarity based on Manhattan distance (higher is more similar)."""
    distance = manhattan_distance(vector_a, vector_b)
    # Convert distance to similarity (0-1 scale, higher is more similar)
    return 1 / (1 + distance)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, store_metadata: bool = True):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.store_metadata = store_metadata
        self.created_at = time.time()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        """Insert a vector with optional metadata"""
        self.vectors[key] = vector
        if self.store_metadata and metadata:
            self.metadata[key] = {
                **metadata,
                "inserted_at": time.time(),
                "vector_dimension": len(vector)
            }
        elif self.store_metadata:
            self.metadata[key] = {
                "inserted_at": time.time(),
                "vector_dimension": len(vector)
            }

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        include_metadata: bool = False,
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        results = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        if include_metadata and self.store_metadata:
            return [(key, score, self.metadata.get(key, {})) for key, score in results]
        
        return results

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
