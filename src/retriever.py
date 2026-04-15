from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    """
    Retrieves and ranks documents based on relevance to the event.
    Combines BM25 and vector-based retrieval using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, top_k: int = 10,
                 use_full_content: bool = False,
                 use_gpu: bool = False,
                 rrf_k: int = 60,
                 use_per_option: bool = False):

        self.top_k = top_k
        self.use_full_content = use_full_content
        self.use_gpu = use_gpu
        self.rrf_k = rrf_k
        self.use_per_option = use_per_option
        
        # Initialize the model
        try:
            model_name = 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer(model_name)
            if use_gpu:
                try:
                    self.model = self.model.to('cuda')
                    print(f"Using GPU for semantic retrieval")
                except:
                    print(f"GPU not available, using CPU")
            print(f"Loaded semantic retrieval model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")


    
    def _retrieve_bm25(self, query: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Retrieve all documents using BM25 with scores.
        Returns list of documents sorted by score descending.
        """    
        try:
            # Preprocessing document/snippet
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            tokenized_texts = [item.lower().split(" ") for item in texts_to_index]
            tokenized_query = query.lower().split(" ")
            
            # Retrieve
            bm25 = BM25Okapi(tokenized_texts)
            scores = bm25.get_scores(tokenized_query)
            sorted_indices = np.argsort(scores)[::-1]
            
            results = [documents[i] for i in sorted_indices]
            return results
           
        except Exception as e:
            print(f"Warning: BM25 retrieval failed ({e}).")
            return None
        

    def _retrieve_semantic(self, query: str, title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Semantic retriever using vector embeddings (sentence transformers).
        Returns list of documents sorted by similarity descending.
        """
        try:
            if self.use_full_content:
                texts_to_index = documents
            else:
                texts_to_index = title_snippet
            
            # Encode query and documents
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            doc_embeddings = self.model.encode(
                texts_to_index,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True,
            )

            # Calculate cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

            # Sort all documents by similarity descending
            sorted_indices = np.argsort(similarities)[::-1]

            results = [documents[i] for i in sorted_indices]
            return results
            
        except Exception as e:
            print(f"Warning: Vector retrieval failed ({e})")
            return None
        
    
    def _rrf_merge(self, bm25_results: List[str], 
                   vector_results: List[str]) -> List[Tuple[str, float]]:
        """
        Merge results from BM25 and vector retrieval using Reciprocal Rank Fusion (RRF).
        RRF score: 1 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)

        # Process vector results
        for rank, doc in enumerate(vector_results, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending
        merged_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged_results[:self.top_k]
    

    def retrieve(self, event: str, title_snippet: List[str], documents: List[str], options: List[str] = None) -> List[str]:
        """
        Retrieve top_k documents using combined BM25 and vector retrieval with RRF.
        If use_per_option is True and options are provided, use per-option retrieval.
        """
        # If per-option is True and options are provided, use this new method
        if self.use_per_option and options:
            return self.retrieve_with_options(event, options, title_snippet, documents)
        
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
    
        # Get results from both methods (all documents ranked)
        bm25_results = self._retrieve_bm25(event, title_snippet, documents)
        vector_results = self._retrieve_semantic(event, title_snippet, documents)

        if not vector_results:
            if not bm25_results:
                return documents
            else:
                return [doc for doc in bm25_results[:self.top_k]]
            
        # Merge using RRF
        merged_results = self._rrf_merge(bm25_results, vector_results)
        return [doc for doc, _ in merged_results]
    

    def retrieve_with_options(self, event: str, options: List[str],
                              title_snippet: List[str], documents: List[str]) -> List[str]:
        """
        Retrieve event + options related documents
        Event's weight 2x，Option's weight 1x (BM25 + Semantic)
        """
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            return documents
        
        all_scores: Dict[str, float] = {}
        
        # 1. Event related (weight 2x)
        bm25_event = self._retrieve_bm25(event, title_snippet, documents)
        vec_event = self._retrieve_semantic(event, title_snippet, documents)
        
        if bm25_event:
            for rank, doc in enumerate(bm25_event, 1):
                all_scores[doc] = all_scores.get(doc, 0) + 2.0 / (self.rrf_k + rank)
        if vec_event:
            for rank, doc in enumerate(vec_event, 1):
                all_scores[doc] = all_scores.get(doc, 0) + 2.0 / (self.rrf_k + rank)
        
        # 2. each option related (weight 1x, BM25 + Semantic)
        for option in options:
            bm25_opt = self._retrieve_bm25(option, title_snippet, documents)
            vec_opt = self._retrieve_semantic(option, title_snippet, documents)
            
            if bm25_opt:
                for rank, doc in enumerate(bm25_opt, 1):
                    all_scores[doc] = all_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)
            if vec_opt:
                for rank, doc in enumerate(vec_opt, 1):
                    all_scores[doc] = all_scores.get(doc, 0) + 1.0 / (self.rrf_k + rank)
        
        # return top_k
        sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:self.top_k]]