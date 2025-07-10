from core.imgdata.storage_manager import StorageManager
from memory.embedding_handler import EmbeddingHandler

class VectorStorageManager(StorageManager):
    """
    Extends StorageManager to also handle vector DB insertion for embeddings and other fields.
    """
    def __init__(self, base_dir, vector_db, embedding_handler=None, config=None):
        super().__init__(base_dir, config)
        self.vector_db = vector_db
        self.embedding_handler = embedding_handler or EmbeddingHandler()

    def save(self, obj, obj_id):
        super().save(obj, obj_id)
        # Compute embedding and insert into vector DB
        if hasattr(obj, 'image') and obj.image is not None:
            embedding = self.embedding_handler.get_embedding(obj.image)
            # Insert embedding and other fields into vector DB
            record = {
                'id': obj_id,
                'embedding': embedding,
                'metadata': getattr(obj, 'metadata', {}),
                'storage_dict': getattr(obj, 'storage_dict', {})
            }
            self.vector_db.insert(record)

    def load(self, cls, obj_id):
        # Optionally, could retrieve from vector DB as well
        return super().load(cls, obj_id)
