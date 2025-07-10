from core.memory.embedding_handler import EmbeddingHandler
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
from core.imgdata.image_data import ImageParseUnit, ImageParseResult


class CustomChromaDB:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "img_parse_units",
        embedding_handler: Optional[EmbeddingHandler] = None,
    ):
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory or ".chroma_db",
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_handler = embedding_handler or EmbeddingHandler()

    def insert_unit(
        self, unit: ImageParseUnit, embedding: Optional[List[float]] = None
    ):
        """
        插入单个 ImageParseUnit 到 ChromaDB。
        embedding: 需外部提供的向量（如CLIP等模型输出），如未提供则自动用 EmbeddingHandler 生成。
        """
        record = unit.to_vector_record()
        uid = unit.get_uid()
        if embedding is None:
            if unit.bbox_image is not None:
                img = unit.bbox_image
            elif unit.image is not None:
                img = unit.get_bbox_image()
            else:
                raise ValueError("No image data available for embedding generation.")
            embedding = self.embedding_handler.get_embedding(img)
        self.collection.add(
            ids=[uid],
            embeddings=[embedding],
            metadatas=[record],
            documents=[unit.text or ""],
        )
        unit.storage_dict["vector_id"] = uid

    def insert_units(
        self,
        units: List[ImageParseUnit],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """
        批量插入多个 ImageParseUnit。
        """
        records = [u.to_vector_record() for u in units]
        uids = [u.get_uid() for u in units]
        texts = [u.text or "" for u in units]
        if embeddings is None:
            imgs = []
            for u in units:
                if u.bbox_image is not None:
                    imgs.append(u.bbox_image)
                elif u.image is not None:
                    imgs.append(u.get_bbox_image())
                else:
                    raise ValueError(f"No image data for unit {u.get_uid()}.")
            embeddings = [self.embedding_handler.get_embedding(img) for img in imgs]
        self.collection.add(
            ids=uids, embeddings=embeddings, metadatas=records, documents=texts
        )
        for u in units:
            u.storage_dict["vector_id"] = u.get_uid()

    def insert_result(
        self,
        result: ImageParseResult,
        embeddings: Optional[List[List[float]]] = None,
        result_embedding: Optional[List[float]] = None,
    ):
        """
        插入 ImageParseResult 及其所有 unit。
        embeddings: 与 result.units 顺序一致的向量列表。未提供则自动生成。
        result_embedding: result 本身的向量（如有，可用聚合/全局特征），未提供则自动生成。
        """
        self.insert_units(result.units, embeddings)
        # 插入 result 本身
        result_uid = result.get_uid()
        record = result.to_vector_records()
        if result_embedding is None:
            # 用全图 embedding
            result_embedding = self.embedding_handler.get_embedding(result.image)
        self.collection.add(
            ids=[result_uid],
            embeddings=[result_embedding],
            metadatas=[record],
            documents=[result.summary_text or ""],
        )
        result.storage_dict["vector_id"] = result_uid

    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        基于向量检索，返回 top_k 个最相似的 unit 元数据。
        """
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        # 返回 metadatas 列表
        return results.get("metadatas", [])

    def persist(self):
        self.client.persist()
