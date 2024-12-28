import os
import os.path as osp
import torch
import time
from typing import List
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from mmengine import load, dump, isfile
from utils import dump_to_json_file


class MySentenceTransformer:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        root_cache_path: str = '',
    ):

        self.model_name = model_name
        self.cache_path = osp.join(root_cache_path,
                                   model_name.rstrip(os.sep) + '.pkl')

        self._load_cache()
        self._load_model()

    def _load_model(self):

        self.model = SentenceTransformer(self.model_name)

    def _load_cache(self):
        self.init_cache_size = 0
        if isfile(self.cache_path):
            print(
                f"loading Sentence Embedding from cache file: {self.cache_path} ..."
            )
            start_time = time.time()
            self.cache_dict = load(self.cache_path)
            end_time = time.time()
            self.init_cache_size = len(self.cache_dict)
            print(
                f"loaded cache size: {self.init_cache_size}, using time: {end_time - start_time} s."
            )
        else:
            self.cache_dict = {}

    def _save_cache(self):

        if self.init_cache_size >= len(self.cache_dict):
            return

        print(
            f"saving Sentence Embedding to cache file: {self.cache_path} ...")
        try:
            start_time = time.time()
            dump(self.cache_dict, self.cache_path)
            end_time = time.time()
            print(
                f"saved cache size: {len(self.cache_dict)}, using time: {end_time - start_time} s."
            )

            start_time = time.time()
            cache_keys = list(self.cache_dict.keys())
            cache_keys.sort()
            cache_keys_file = self.cache_path.replace('.pkl', '.json')
            dump_to_json_file(cache_keys, cache_keys_file)
            end_time = time.time()
            print(
                f"saved cache keys to file: {cache_keys_file}, using time: {end_time - start_time} s."
            )

        except Exception as e:
            print(f"save cache failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_cache()

    def similarity(self, embeddings1: List[float], embeddings2: List[float]):
        # 将列表转换为 Tensor
        embeddings1 = torch.tensor(embeddings1)
        embeddings2 = torch.tensor(embeddings2)
        return cosine_similarity(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=-1)
    

    def encode(self, sentence: str):

        if sentence in self.cache_dict:
            return self.cache_dict[sentence]

        embedding = self.model.encode(sentence)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        self.cache_dict[sentence] = embedding

        return embedding

    def encodes(self, sentences: List[str]):
        embeddings = []
        for sentence in sentences:
            embeddings.append(self.encode(sentence))

        return embeddings
