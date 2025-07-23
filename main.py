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
        self.bengali_stopwords = set([
            'এবং', 'বা', 'যে', 'এই', 'সে', 'তার', 'তাকে', 'তাদের', 'আমি', 'আমার', 
            'আমাদের', 'তুমি', 'তোমার', 'তোমাদের', 'সেই', 'এটি', 'এটা', 'ওটা', 
            'কিন্তু', 'তবে', 'যদি', 'যখন', 'কেন', 'কিভাবে', 'কোথায়', 'কে', 'কি',
            'হয়', 'হয়েছে', 'হবে', 'ছিল', 'থাকে', 'আছে', 'নেই', 'না', 'নয়',
            'দিয়ে', 'করে', 'হয়ে', 'গিয়ে', 'এসে', 'থেকে', 'পর্যন্ত', 'মধ্যে'
        ])
        
        self.bengali_punctuation = '।,;:!?""''()[]{}–—…'
    
    def clean_bengali_text(self, text: str) -> str:
        """Clean and normalize Bengali text"""
        if not text:
            return ""
        
        text = unicodedata.normalize('NFC', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\u0980-\u09FF\u0020\u002E\u002C\u003B\u003A\u0021\u003F\u201C\u201D\u2018\u2019\u0028\u0029\u005B\u005D\u007B\u007D\u2013\u2014\u2026]', ' ', text)
        
        text = re.sub(r'[।]', '।', text)  
        
        return text.strip()
    
    def tokenize_bengali(self, text: str) -> List[str]:
        """Tokenize Bengali text into words"""
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
        
        sentences = re.split(r'[।\.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_word_count = len(words)
            
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
        
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  
        
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
    
    
    
class ConversationMemory:
    """Manage both short-term (conversation) and long-term (document) memory"""
    
    def __init__(self, max_short_term: int = 10):
        self.short_term_memory = []  
        self.max_short_term = max_short_term
        self.long_term_memory = None  
        
    def add_to_short_term(self, query: str, response: str):
        """Add query-response pair to short-term memory"""
        self.short_term_memory.append({
            'timestamp': datetime.now(),
            'query': query,
            'response': response
        })
        
        
        if len(self.short_term_memory) > self.max_short_term:
            self.short_term_memory = self.short_term_memory[-self.max_short_term:]
    
    def get_context(self, current_query: str) -> str:
        """Get relevant context from short-term memory"""
        if not self.short_term_memory:
            return ""
        
        context = "Previous conversation:\n"
        for memory in self.short_term_memory[-3:]:  
            context += f"Q: {memory['query']}\nA: {memory['response']}\n\n"
        
        return context
    
    def set_long_term_memory(self, vector_store: MultilingualVectorStore):
        """Set the document vector store as long-term memory"""
        self.long_term_memory = vector_store
    
    
    
    ##TESTING##
    
    
    

if __name__ == "__main__":
    
    memory = ConversationMemory(max_short_term=5)

    
    memory.add_to_short_term("তুমি কে?", "আমি একটি এআই সহকারী।")
    memory.add_to_short_term("বাংলা কি তোমার মাতৃভাষা?", "না, তবে আমি বাংলা বুঝতে পারি।")
    memory.add_to_short_term("তুমি ইংরেজি বলো?", "হ্যাঁ, আমি ইংরেজিতেও কথা বলতে পারি।")
    memory.add_to_short_term("তোমার নাম কী?", "আমার কোনও নির্দিষ্ট নাম নেই।")
    memory.add_to_short_term("তুমি কিভাবে কাজ করো?", "আমি মেশিন লার্নিং মডেল হিসেবে কাজ করি।")
    memory.add_to_short_term("তুমি কী জানতে পারো?", "আমি বিভিন্ন তথ্য দিতে পারি।")

    
    print("=== Short-Term Memory Context ===")
    context = memory.get_context("তুমি কিভাবে কাজ করো?")
    print(context)

   
    dummy_vector_store = MultilingualVectorStore()
    memory.set_long_term_memory(dummy_vector_store)
    print("Long-term memory set:", isinstance(memory.long_term_memory, MultilingualVectorStore))