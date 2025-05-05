from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_community.vectorstores import Chroma

import xc_config

checkpoint_path = str(xc_config._xc_checkpoints_dir / "laion2b_s34b_b79k/open_clip_model.safetensors")

class MultiEmbeddingFunction(OpenCLIPEmbeddingFunction):
    def __init__(self, model_name = "ViT-B-32", checkpoint = checkpoint_path, device = "cuda"):
        super().__init__(model_name, checkpoint, device)

    def embed_image(self, uris: list[str]) -> list:
        from image_entity import MultiFormatImage
        res = []
        for uri in uris:
            image = MultiFormatImage.load(uri).get('numpy')
            res.append(self._encode_image(image))
        return res

embedding_function = MultiEmbeddingFunction()

class ChromadbVectorStore(Chroma):
    def __init__(self,
                 collection_name = "multimodal_collection",
                 embedding_function = embedding_function,
                 persist_directory = xc_config._xc_memory_data_dir,
                 client_settings = None,
                 collection_metadata = None,
                 client = None,
                 relevance_score_fn = None):
        super().__init__(collection_name, embedding_function, persist_directory, client_settings, collection_metadata, client, relevance_score_fn)

    def add_image_item(self, image_entity):
        id, metadata, uri = image_entity.get_chromadb_item()
        self.add_images(ids=[id], metadatas=[metadata], uris=[uri])

    def query_uri_item(self, uri):
        return self.similarity_search_by_image(uri, 1)

    def query_image_item(self, image):
        return self._collection.query(query_images=image)

if __name__ == "__main__":
    uri = r'E:\xingchen\chroma_db\images\2025-05-05_18-49-22.918.png'
    # from image_entity import MultiFormatImage
    # image = MultiFormatImage.load(uri).get()

    test = ChromadbVectorStore()
    result = test.query_uri_item(uri)
    # result = test.query_image_item(image)
    print(len(result))
