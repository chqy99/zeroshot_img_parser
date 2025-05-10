import chromadb
import os
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

import xc_config
from image_entity import ImageAnnotation

checkpoint_path = str(xc_config._xc_checkpoints_dir / "laion2b_s34b_b79k/open_clip_model.safetensors")
embedding_function = OpenCLIPEmbeddingFunction(checkpoint=checkpoint_path, device="cuda")
data_loader = ImageLoader()

class ChromadbVectorStore:
    def __init__(self, collection_name: str = "multimodal_collection", persist_directory: str = xc_config._xc_memory_data_dir):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = collection_name
        self._load_or_create_collection()

    def _load_or_create_collection(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name,
                                                                   embedding_function=embedding_function,
                                                                   data_loader=data_loader)
        except Exception as e:
            raise

    # 按掩膜保存，or按图像保存
    def add_image_item(self, image_entity: ImageAnnotation):
        id, metadata, image = image_entity.get_chromadb_item()
        if len(self.collection.get(ids=id)["ids"]) > 0:
            self.collection.update(ids=id,
                               metadatas=metadata,
                               images=image)
        else:
            self.collection.add(ids=id,
                                metadatas=metadata,
                                images=image)

    # 截图后，先查询有无相关记录
    def query_image_item(self, image, distance_limit=0.03) -> ImageAnnotation:
        print(self.collection.count())
        res = self.collection.query(query_images=image, n_results=1)
        print(res)
        if len(res["distances"][0]) > 0 and res["distances"][0][0] <= distance_limit:
            return ImageAnnotation(image=image, id=res["ids"][0][0], metadata=res["metadatas"][0][0])
        else:
            return ImageAnnotation(image=image)