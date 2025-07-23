import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import os
import re
import json
import pickle
from typing import List, Dict, Tuple, Any
import numpy as np
from datetime import datetime
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
import fitz  
import re
import unicodedata
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class BengaliTextProcessor:
    """Specialized processor for Bengali text cleaning and normalization"""
    
    def __init__(self):
        # Bengali stopwords (common words to filter out)
        self.bengali_stopwords = set([
            'এবং', 'বা', 'যে', 'এই', 'সে', 'তার', 'তাকে', 'তাদের', 'আমি', 'আমার', 
            'আমাদের', 'তুমি', 'তোমার', 'তোমাদের', 'সেই', 'এটি', 'এটা', 'ওটা', 
            'কিন্তু', 'তবে', 'যদি', 'যখন', 'কেন', 'কিভাবে', 'কোথায়', 'কে', 'কি',
            'হয়', 'হয়েছে', 'হবে', 'ছিল', 'থাকে', 'আছে', 'নেই', 'না', 'নয়',
            'দিয়ে', 'করে', 'হয়ে', 'গিয়ে', 'এসে', 'থেকে', 'পর্যন্ত', 'মধ্যে'
        ])
        
        # Bengali punctuation marks
        self.bengali_punctuation = '।,;:!?""''()[]{}–—…'
    
    def clean_bengali_text(self, text: str) -> str:
        """Clean and normalize Bengali text"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted characters but keep Bengali punctuation
        text = re.sub(r'[^\u0980-\u09FF\u0020\u002E\u002C\u003B\u003A\u0021\u003F\u201C\u201D\u2018\u2019\u0028\u0029\u005B\u005D\u007B\u007D\u2013\u2014\u2026]', ' ', text)
        
        # Normalize punctuation
        text = re.sub(r'[।]', '।', text)  # Ensure proper Bengali full stop
        
        return text.strip()
    
    def tokenize_bengali(self, text: str) -> List[str]:
        """Tokenize Bengali text into words"""
        # Simple word tokenization for Bengali
        words = re.findall(r'\S+', text)
        return [word for word in words if word not in self.bengali_punctuation]
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove Bengali stopwords"""
        return [word for word in words if word.lower() not in self.bengali_stopwords]
    



class DocumentProcessor:
    """Process PDF documents and extract clean text chunks"""
    
    def __init__(self):
        self.bengali_processor = BengaliTextProcessor()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF for better Unicode support"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            doc.close()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
        
        return text
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Create overlapping text chunks with metadata"""
        # Clean the text
        cleaned_text = self.bengali_processor.clean_bengali_text(text)
        
        # Split into sentences using both English and Bengali sentence endings
        sentences = re.split(r'[।\.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_word_count = len(words)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_words + sentence_word_count > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': len(chunks),
                    'word_count': current_words,
                    'sentence_range': (i - len(current_chunk.split('।')) + 1, i)
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('।')[-2:] if '।' in current_chunk else []
                current_chunk = '।'.join(overlap_sentences) + '।' if overlap_sentences else ""
                current_words = len(current_chunk.split())
            
            current_chunk += sentence + "।"
            current_words += sentence_word_count
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': len(chunks),
                'word_count': current_words,
                'sentence_range': (len(sentences) - 1, len(sentences))
            })
        
        return chunks
    
    
    
class MultilingualVectorStore:
    """Vector store supporting both Bengali and English"""
    
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.bengali_processor = BengaliTextProcessor()
        
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to the vector store"""
        self.chunks = chunks
        
        # Create embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index for efficient similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Added {len(chunks)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        # Clean and process query
        cleaned_query = self.bengali_processor.clean_bengali_text(query)
        
        # Create query embedding
        query_embedding = self.model.encode([cleaned_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    
    
    ##TESTING##
    
    
    
if __name__ == "__main__":
    # Sample chunks (normally generated from PDF using DocumentProcessor)
    chunks = [
        {'text': 'আমি একজন ছাত্র। আমি প্রতিদিন স্কুলে যাই।', 'chunk_id': 0, 'word_count': 8, 'sentence_range': (0, 1)},
        {'text': 'বাংলা ভাষা আমার মাতৃভাষা।', 'chunk_id': 1, 'word_count': 5, 'sentence_range': (2, 3)},
        {'text': 'I love programming and solving problems.', 'chunk_id': 2, 'word_count': 6, 'sentence_range': (4, 5)},
    ]

    # Initialize the vector store
    vector_store = MultilingualVectorStore()

    # Add document chunks
    vector_store.add_documents(chunks)

    # Run a search query
    query = "আমি প্রতিদিন কোথায় যাই?"
    results = vector_store.search(query, k=2)

    # Print the results
    print("\nSearch Results:")
    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(f"Text: {chunk['text']}")
        print("-" * 50)