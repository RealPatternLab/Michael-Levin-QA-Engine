#!/usr/bin/env python3
"""
RAG Query Engine Component
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from webapp.utils.api_keys import get_api_key

class RAGQueryEngine:
    def __init__(self):
        """Initialize the RAG query engine."""
        # Check API key first
        api_key = get_api_key()
        if not api_key or api_key.strip() == '':
            raise ValueError("OpenAI API key not available")
        
        self.client = OpenAI(api_key=api_key)
        self.index = None
        self.chunks = None
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and combined chunks."""
        try:
            # Load FAISS index
            index_path = Path("outputs/combined/faiss_index.bin")
            if not index_path.exists():
                raise FileNotFoundError("FAISS index not found. Run the pipeline first.")
            
            self.index = faiss.read_index(str(index_path))
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load combined chunks
            chunks_path = Path("outputs/combined/combined_chunks.json")
            if not chunks_path.exists():
                raise FileNotFoundError("Combined chunks not found. Run the pipeline first.")
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.chunks)} semantic chunks")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text query."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            embedding = response.data[0].embedding
            return np.array(embedding).astype('float32').reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Search for semantically similar chunks."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Get results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'chunk': chunk
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise 