"""
Legal IPC-RAG — FIR Vector Store
================================
Manages embedding and indexing of processed FIR documents for semantic search.
Uses ChromaDB and Sentence-Transformers.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIRVectorStore:
    """
    Vector Store for FIR narratives and metadata.
    """

    def __init__(self, persist_directory: str = "data/vector_store/fir_chroma"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Using a legal-friendly or robust general embedding model
        # 'all-MiniLM-L6-v2' is fast, 'multi-qa-mpnet-base-dot-v1' is better for Q&A
        self.model_name = "multi-qa-mpnet-base-dot-v1"
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="fir_collection",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_fir(self, fir_doc: Dict):
        """Add a single structured FIR document to the store."""
        narrative = fir_doc.get("narrative", "")
        if not narrative or len(narrative) < 50:
            return

        # Prepare metadata
        metadata = {
            "fir_number": fir_doc.get("fir_number", "UNKNOWN"),
            "police_station": fir_doc.get("police_station", ""),
            "district": fir_doc.get("district", ""),
            "ipc_sections": ",".join(fir_doc.get("applied_ipc_sections", [])),
            "source": fir_doc.get("processing_metadata", {}).get("source_dataset", "unknown")
        }

        self.collection.add(
            documents=[narrative],
            metadatas=[metadata],
            ids=[f"FIR_{fir_doc.get('fir_number', 'ID').replace('/', '_')}"]
        )

    def batch_add_from_dir(self, directory: str):
        """Recursively add all JSON FIRs from a directory."""
        logger.info(f"Indexing FIRs from {directory}...")
        
        json_files = list(Path(directory).rglob("*.json"))
        
        # Process in batches for efficiency
        batch_size = 100
        for i in tqdm(range(0, len(json_files), batch_size), desc="Indexing Batches"):
            batch_files = json_files[i : i + batch_size]
            
            docs = []
            metas = []
            ids = []
            
            for fpath in batch_files:
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        fir = json.load(f)
                        
                    narrative = fir.get("narrative", "")
                    if not narrative or len(narrative) < 50: continue
                    
                    docs.append(narrative)
                    metas.append({
                        "fir_number": str(fir.get("fir_number", "UNKNOWN")),
                        "police_station": str(fir.get("police_station", "")),
                        "ipc_sections": ",".join(fir.get("applied_ipc_sections", [])),
                        "source": fir.get("processing_metadata", {}).get("source_dataset", "unknown")
                    })
                    ids.append(f"FIR_{str(fir.get('fir_number', i)).replace('/', '_')}_{fpath.stem}")
                except Exception as e:
                    logger.error(f"Error reading {fpath.name}: {e}")
            
            if docs:
                self.collection.add(documents=docs, metadatas=metas, ids=ids)
        
        logger.info(f"Finished indexing. Total count in store: {self.collection.count()}")

    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar FIRs given a query narrative."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    # Demo initialization
    store = FIRVectorStore()
    print(f"Vector Store Initialized with model: {store.model_name}")
    
    # Check if empty, then index processed data
    if store.collection.count() == 0:
        store.batch_add_from_dir("data/processed/fir_processed")
    else:
        print(f"Store already contains {store.collection.count()} documents.")
