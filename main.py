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
    
    
    
    ##TESTING##
    
    
    
if __name__ == "__main__":
    processor = BengaliTextProcessor()
    
    sample_text = "আমি আজ স্কুলে গিয়েছিলাম এবং বন্ধুদের সাথে দেখা করেছি। এটা দারুণ ছিল!"
    
    print("Original Text:")
    print(sample_text)
    
    # Step 1: Clean text
    cleaned = processor.clean_bengali_text(sample_text)
    print("\nCleaned Text:")
    print(cleaned)
    
    # Step 2: Tokenize
    tokens = processor.tokenize_bengali(cleaned)
    print("\nTokens:")
    print(tokens)
    
    # Step 3: Remove stopwords
    filtered_tokens = processor.remove_stopwords(tokens)
    print("\nFiltered Tokens (No Stopwords):")
    print(filtered_tokens)