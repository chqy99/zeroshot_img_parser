import os
import yaml
from typing import Optional, List
from core.imgdata.image_data import ImageParseUnit, ImageParseResult
from core.memory.custom_chromadb import CustomChromaDB

class ImageMemory:
    def __init__(self, config_path: Optional[str] = None):
        # 加载配置
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "memory_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.base_dir = cfg.get("base_dir", "./img_memory")
        self.vector_db_dir = cfg.get("vector_db_dir", "./.chroma_db")
        self.collection_name = cfg.get("collection_name", "img_parse_units")
        self.image_save_fields = cfg.get("image_save_fields", ["bbox_image", "mask_image"])
        self.db = CustomChromaDB(persist_directory=self.vector_db_dir, collection_name=self.collection_name)

    def save_unit(self, unit: ImageParseUnit, embedding: Optional[List[float]] = None):
        # 保存图片
        unit.save_image(self.base_dir, image_filter=self.image_save_fields)
        # 插入向量库
        self.db.insert_unit(unit, embedding=embedding)
        self.db.persist()

    def save_units(self, units: List[ImageParseUnit], embeddings: Optional[List[List[float]]] = None):
        for u in units:
            u.save_image(self.base_dir, image_filter=self.image_save_fields)
        self.db.insert_units(units, embeddings=embeddings)
        self.db.persist()

    def save_result(self, result: ImageParseResult, embeddings: Optional[List[List[float]]] = None, result_embedding: Optional[List[float]] = None):
        # 保存所有 unit 的图片
        for u in result.units:
            u.save_image(self.base_dir, image_filter=self.image_save_fields)
        # 保存 result 级图片（如有）
        result.save_image(self.base_dir, image_filter=["image", "bboxs_image", "masks_image"])
        # 插入向量库
        self.db.insert_result(result, embeddings=embeddings, result_embedding=result_embedding)
        self.db.persist()
