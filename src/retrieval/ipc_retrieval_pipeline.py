"""
Hybrid Retrieval Pipeline (IPC-Specific) — Phase 2.5
Implements multi-mode retrieval for IPC sections and citizen explanations.
Uses Dense (GIST-Law-Embed) + Sparse (BM25) + Cross-Encoder reranking.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
except ImportError:
    torch, SentenceTransformer, util, CrossEncoder = None, None, None, None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


@dataclass
class RetrievalResult:
    query: str
    results: List[Dict[str, Any]]
    retrieval_mode: str

@dataclass
class CitizenContext:
    section_number: str
    title: str
    plain_language_summary: str
    rights_of_accused: str
    bail_conditions: str
    punishment_range: str

@dataclass
class AmendmentHistory:
    section_number: str
    latest_version: str
    recently_amended: bool
    history_notes: str


class IPCHybridRetriever:
    def __init__(self,
                 ipc_kb_path: str = "data/processed/ipc_sections/ipc_complete.json",
                 embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
                 nli_model_name: str = "cross-encoder/nli-deberta-v3-small"):
        
        self.ipc_kb_path = ipc_kb_path
        self.ipc_kb = self._load_ipc_kb()
        
        # Build corpora for search
        self.sections_list = list(self.ipc_kb.values())
        self.section_texts = [
            f"Section {s.get('section_number', '')}: {s.get('title', '')}. {s.get('full_text', '')}"
            for s in self.sections_list
        ]
        self.section_numbers = [str(s.get('section_number', '')) for s in self.sections_list]

        print(f"Loading embedding model '{embedding_model_name}'...")
        if SentenceTransformer:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            # Pre-compute embeddings for dense search
            print("Pre-computing IPC section embeddings...")
            if self.section_texts:
                self.section_embeddings = self.embedding_model.encode(self.section_texts, convert_to_tensor=True)
            else:
                self.section_embeddings = None
        else:
            self.embedding_model = None
            self.section_embeddings = None

        print(f"Loading NLI cross-encoder '{nli_model_name}' for reranking...")
        if CrossEncoder:
            self.cross_encoder = CrossEncoder(nli_model_name)
        else:
            self.cross_encoder = None

        print("Initializing BM25 index...")
        if BM25Okapi and self.section_texts:
            # Simple tokenization for BM25
            tokenized_corpus = [doc.lower().split() for doc in self.section_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def _load_ipc_kb(self) -> Dict[str, Any]:
        """Load the structured IPC knowledge base."""
        if not Path(self.ipc_kb_path).exists():
            print(f"Warning: IPC KB not found at {self.ipc_kb_path}")
            return {}
        with open(self.ipc_kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {str(item.get("section_number")): item for item in data if item.get("section_number")}
            elif isinstance(data, dict):
                return data
        return {}

    def retrieve_ipc_context(self,
                             query: str,
                             section_numbers: Optional[List[str]] = None,
                             retrieval_mode: str = 'hybrid') -> RetrievalResult:
        """
        Three retrieval modes:
        - Mode A (section_specific): Direct lookup.
        - Mode B (semantic): Dense vector search.
        - Mode C (hybrid): BM25 + Dense + Cross-Encoder Rerank.
        """
        results = []

        if retrieval_mode == 'section_specific' or section_numbers:
            # Mode A: Section-Specific Retrieval
            if section_numbers:
                for sec in section_numbers:
                    sec_str = str(sec)
                    if sec_str in self.ipc_kb:
                        item = dict(self.ipc_kb[sec_str])
                        item['source'] = 'IPC_SECTION'
                        item['retrieval_score'] = 1.0
                        results.append(item)
            return RetrievalResult(query=query, results=results, retrieval_mode='section_specific')

        elif retrieval_mode == 'semantic':
            # Mode B: Semantic Retrieval
            if self.embedding_model and self.section_embeddings is not None and torch:
                query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, self.section_embeddings)[0]
                top_results = torch.topk(cosine_scores, k=min(10, len(self.sections_list)))
                
                for score, idx in zip(top_results[0], top_results[1]):
                    item = dict(self.sections_list[idx])
                    item['source'] = 'SEMANTIC_SEARCH'
                    item['retrieval_score'] = float(score.item())
                    results.append(item)
            return RetrievalResult(query=query, results=results, retrieval_mode='semantic')

        else:
            # Mode C: Hybrid (DEFAULT)
            # 1. Semantic Search (Top 15)
            semantic_candidates = {}
            if self.embedding_model and self.section_embeddings is not None and torch:
                query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, self.section_embeddings)[0]
                top_results = torch.topk(cosine_scores, k=min(15, len(self.sections_list)))
                for rank, idx in enumerate(top_results[1].tolist()):
                    sec_num = self.section_numbers[idx]
                    semantic_candidates[sec_num] = rank + 1

            # 2. BM25 Search (Top 15)
            bm25_candidates = {}
            if self.bm25:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]
                for rank, idx in enumerate(top_indices):
                    sec_num = self.section_numbers[idx]
                    bm25_candidates[sec_num] = rank + 1

            # 3. Reciprocal Rank Fusion (RRF)
            k = 60
            rrf_scores = {}
            all_candidate_secs = set(list(semantic_candidates.keys()) + list(bm25_candidates.keys()))
            
            for sec_num in all_candidate_secs:
                score = 0.0
                if sec_num in semantic_candidates:
                    score += 1.0 / (k + semantic_candidates[sec_num])
                if sec_num in bm25_candidates:
                    score += 1.0 / (k + bm25_candidates[sec_num])
                rrf_scores[sec_num] = score

            # Get Top 10 fused candidates
            top_fused = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]

            # 4. Cross-Encoder Reranking
            if self.cross_encoder and top_fused:
                pairs = []
                fused_items = []
                for sec_num in top_fused:
                    item = self.ipc_kb[sec_num]
                    text = f"Section {item.get('section_number')}: {item.get('title')}. {item.get('full_text')}"
                    pairs.append([query, text])
                    fused_items.append(item)

                # Predict scores (higher is better entailment/relevance)
                # Since we're using NLI cross-encoder, index 2 is entailment
                ce_scores = self.cross_encoder.predict(pairs)
                entailment_scores = [float(s[2]) if isinstance(s, (list, tuple)) or hasattr(s, '__iter__') else float(s) for s in ce_scores]

                # Sort by entailment
                reranked = sorted(zip(fused_items, entailment_scores), key=lambda x: x[1], reverse=True)
                
                # Take top 5 with diversity filtering (simplified here to just top 5)
                for item, score in reranked[:5]:
                    result_item = dict(item)
                    result_item['source'] = 'HYBRID_RERANKED'
                    result_item['retrieval_score'] = score
                    results.append(result_item)
            else:
                # Fallback to RRF top 5 if no cross-encoder
                for sec_num in top_fused[:5]:
                    result_item = dict(self.ipc_kb[sec_num])
                    result_item['source'] = 'HYBRID_RRF'
                    result_item['retrieval_score'] = rrf_scores[sec_num]
                    results.append(result_item)

            return RetrievalResult(query=query, results=results, retrieval_mode='hybrid')

    def retrieve_citizen_explanation_context(self, section_number: str) -> CitizenContext:
        """Special retrieval for citizen-friendly explanations."""
        if section_number not in self.ipc_kb:
            return CitizenContext(
                section_number=section_number,
                title="Unknown",
                plain_language_summary="Section not found in KB.",
                rights_of_accused="Unknown",
                bail_conditions="Unknown",
                punishment_range="Unknown"
            )
            
        data = self.ipc_kb[section_number]
        
        # In a full system, this would retrieve from a specific "citizen guides" DB.
        # Here we synthesize it from the KB data.
        bailable = data.get("bailable", False)
        bail_cond = "Bailable as a matter of right. Police or Court must grant bail." if bailable else "Non-bailable. Bail is at the discretion of the Court."
        
        return CitizenContext(
            section_number=section_number,
            title=data.get("title", ""),
            plain_language_summary=f"This section deals with {data.get('title', '').lower()}.",
            rights_of_accused="You have the right to legal counsel and to remain silent.",
            bail_conditions=bail_cond,
            punishment_range=data.get("punishment", data.get("maximum_punishment", "Determined by court."))
        )

    def add_temporal_context(self, section_number: str) -> AmendmentHistory:
        """Retrieve amendment history for IPC section."""
        # Note: In 2023/2024, the IPC is replaced by Bharatiya Nyaya Sanhita (BNS).
        # We flag sections that might be affected.
        
        history_notes = "Under the Bharatiya Nyaya Sanhita (BNS) 2023, this IPC section has been reorganized or modified."
        
        return AmendmentHistory(
            section_number=section_number,
            latest_version="Pre-2023 IPC",
            recently_amended=True, # Flagging as True due to BNS transition
            history_notes=history_notes
        )


# Standalone testing
if __name__ == "__main__":
    retriever = IPCHybridRetriever()
    
    query = "Theft of a valuable item from someone's house"
    print(f"\n--- Testing Hybrid Retrieval for query: '{query}' ---")
    
    result = retriever.retrieve_ipc_context(query, retrieval_mode='hybrid')
    
    for i, res in enumerate(result.results):
        print(f"\nRank {i+1}: Section {res['section_number']} - {res['title']}")
        print(f"Score: {res['retrieval_score']:.4f}")
        print(f"Source: {res['source']}")
        
    print("\n--- Testing Citizen Context for Section 378 ---")
    cit_ctx = retriever.retrieve_citizen_explanation_context("378")
    print(json.dumps(asdict(cit_ctx), indent=2))
