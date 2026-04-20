import logging
from typing import List, Tuple, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    文档检索器：
    - 稀疏检索：BM25
    - 稠密检索：SentenceTransformer 向量相似度
    - 排序融合：RRF（Reciprocal Rank Fusion）
    """

    def __init__(
        self,
        top_k: int = 10,
        use_full_content: bool = False,
        use_gpu: bool = False,
        rrf_k: int = 60,
        use_per_option: bool = False,
    ):
        self.top_k = top_k
        self.use_full_content = use_full_content
        self.use_gpu = use_gpu
        self.rrf_k = rrf_k
        self.use_per_option = use_per_option

        # 语义检索模型初始化
        model_name = "all-MiniLM-L6-v2"
        try:
            self.model = SentenceTransformer(model_name)
            if use_gpu:
                try:
                    self.model = self.model.to("cuda")
                    logger.info("Using GPU for semantic retrieval")
                except Exception:
                    logger.warning("GPU not available, falling back to CPU")
            logger.info("Loaded semantic retrieval model: %s", model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def _get_indexing_texts(
        self, title_snippet: List[str], documents: List[str]
    ) -> List[str]:
        """根据配置决定使用全文或 title+snippet 作为检索语料。"""
        return documents if self.use_full_content else title_snippet

    def _retrieve_bm25(
        self, query: str, title_snippet: List[str], documents: List[str]
    ) -> List[int]:
        """
        BM25 排序，返回文档下标（按分数降序）。
        """
        try:
            texts = self._get_indexing_texts(title_snippet, documents)
            tokenized_texts = [t.lower().split() for t in texts]
            tokenized_query = query.lower().split()

            bm25 = BM25Okapi(tokenized_texts)
            scores = bm25.get_scores(tokenized_query)
            return np.argsort(scores)[::-1].tolist()

        except Exception as e:
            logger.warning("BM25 retrieval failed: %s", e)
            return None

    def _retrieve_semantic(
        self, query: str, title_snippet: List[str], documents: List[str]
    ) -> List[int]:
        """
        向量语义排序，返回文档下标（按相似度降序）。
        """
        try:
            texts = self._get_indexing_texts(title_snippet, documents)

            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            doc_embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True,
            )

            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
            return np.argsort(similarities)[::-1].tolist()

        except Exception as e:
            logger.warning("Semantic retrieval failed: %s", e)
            return None

    def _rrf_merge(
        self,
        bm25_ranking: List[int],
        semantic_ranking: List[int],
        num_docs: int,
    ) -> List[int]:
        """
        使用 RRF 融合 BM25 与语义检索结果。
        融合对象是“文档下标”而不是文本，避免文本去重冲突。
        """
        rrf_scores = [0.0] * num_docs

        for rank, idx in enumerate(bm25_ranking, 1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

        for rank, idx in enumerate(semantic_ranking, 1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

        scored = [(idx, score) for idx, score in enumerate(rrf_scores) if score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored[: self.top_k]]

    def retrieve(
        self,
        event: str,
        title_snippet: List[str],
        documents: List[str],
        options: List[str] = None,
    ) -> List[str]:
        """
        标准检索入口：
        - 默认：事件查询 + 混合检索 + RRF
        - 可选：use_per_option 开启后切换到“事件+选项加权检索”
        """
        if self.use_per_option and options:
            return self.retrieve_with_options(
                event, options, title_snippet, documents
            )

        if not documents:
            return []

        # 文档总量不超过 top_k 时直接全返回，减少无意义计算
        if len(documents) <= self.top_k:
            return documents

        bm25_ranking = self._retrieve_bm25(event, title_snippet, documents)
        semantic_ranking = self._retrieve_semantic(event, title_snippet, documents)

        if not semantic_ranking:
            if not bm25_ranking:
                return documents
            return [documents[i] for i in bm25_ranking[: self.top_k]]

        merged_indices = self._rrf_merge(
            bm25_ranking, semantic_ranking, len(documents)
        )
        return [documents[i] for i in merged_indices]

    def retrieve_with_options(
        self,
        event: str,
        options: List[str],
        title_snippet: List[str],
        documents: List[str],
    ) -> List[str]:
        """
        事件+选项加权检索：
        - 事件相关性权重 2x
        - 每个选项相关性权重 1x
        适合多选因果题，能提升对候选项的证据覆盖。
        """
        if not documents:
            return []

        if len(documents) <= self.top_k:
            return documents

        num_docs = len(documents)
        all_scores = [0.0] * num_docs

        # 事件查询部分（权重 2x）
        bm25_event = self._retrieve_bm25(event, title_snippet, documents)
        vec_event = self._retrieve_semantic(event, title_snippet, documents)

        if bm25_event:
            for rank, idx in enumerate(bm25_event, 1):
                all_scores[idx] += 2.0 / (self.rrf_k + rank)
        if vec_event:
            for rank, idx in enumerate(vec_event, 1):
                all_scores[idx] += 2.0 / (self.rrf_k + rank)

        # 选项查询部分（每项权重 1x）
        for option in options:
            bm25_opt = self._retrieve_bm25(option, title_snippet, documents)
            vec_opt = self._retrieve_semantic(option, title_snippet, documents)

            if bm25_opt:
                for rank, idx in enumerate(bm25_opt, 1):
                    all_scores[idx] += 1.0 / (self.rrf_k + rank)
            if vec_opt:
                for rank, idx in enumerate(vec_opt, 1):
                    all_scores[idx] += 1.0 / (self.rrf_k + rank)

        scored = [(idx, score) for idx, score in enumerate(all_scores) if score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [documents[idx] for idx, _ in scored[: self.top_k]]
