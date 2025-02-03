from chromadb import PersistentClient, Settings
from sentence_transformers import SentenceTransformer

class MemorySystem:
    def __init__(self):
        self.encoder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
        self.client = PersistentClient(
            path="data/memories",
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection("conversation_memories")

    def store_memory(self, user_id, query, response):
        embedding = self.encoder.encode(f"{query} {response}").tolist()
        self.collection.add(
            documents=f"{query}\n{response}",
            embeddings=[embedding],
            metadatas={"user_id": user_id},
            ids=str(hash(f"{user_id}{query}"))
        )

    def retrieve_context(self, query, user_id, top_k=3):
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        return "\n".join(results['documents'][0])