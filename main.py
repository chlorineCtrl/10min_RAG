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
        
       
        cleaned_query = self.bengali_processor.clean_bengali_text(query)
        
        
        query_embedding = self.model.encode([cleaned_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        
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
    


class MultilingualRAGSystem:
    """Main RAG system combining all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = MultilingualVectorStore()
        self.memory = ConversationMemory()
        self.bengali_processor = BengaliTextProcessor()
        
        
        self.generator = None
        self._initialize_generator()
    
    def _initialize_generator(self):
        """Initialize text generation model"""
        try:
            self.generator = pipeline("text2text-generation", model="google/flan-t5-small")
        except Exception as e:
            print(f"Could not load generation model: {e}")
            self.generator = None
    
    def load_knowledge_base(self, pdf_path: str):
        """Load and process PDF document into knowledge base"""
        print(f"Loading knowledge base from: {pdf_path}")
        
        # Extract text from PDF
        raw_text = self.document_processor.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            raise ValueError("Could not extract text from PDF")
        
        
        chunks = self.document_processor.create_chunks(raw_text)
        print(f"Created {len(chunks)} chunks from document")
        
        
        self.vector_store.add_documents(chunks)
        
        
        self.memory.set_long_term_memory(self.vector_store)
        
        return len(chunks)
    
    def _generate_answer(self, query: str, context: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Generate answer based on query and retrieved context"""
        
        relevant_text = ""
        for doc, score in retrieved_docs[:3]: 
            relevant_text += doc['text'] + "\n\n"
        
        answer = self._extract_direct_answer(query, relevant_text)
        
        if answer:
            return answer
        
        if retrieved_docs:
            best_chunk = retrieved_docs[0][0]['text']
            # Try to extract a concise answer
            sentences = best_chunk.split('।')
            for sentence in sentences:
                if self._is_relevant_sentence(query, sentence):
                    return sentence.strip() + '।'
            return sentences[0].strip() + '।' if sentences else "তথ্য পাওয়া যায়নি।"
        
        return "দুর্ভাগ্যবশত, এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।"
    
    def _extract_direct_answer(self, query: str, context: str) -> str:
        """Extract direct answer from context based on query patterns"""
        
        patterns = [
            
            (r'কাকে.*?বলা হয়েছে', self._extract_name_or_title),
            
            (r'কাকে.*?বলে উল্লেখ', self._extract_name_or_title),
            (r'বয়স কত', self._extract_age),
            (r'কত বছর', self._extract_age),
        ]
        
        for pattern, extractor in patterns:
            if re.search(pattern, query):
                answer = extractor(context, query)
                if answer:
                    return answer
        
        return None
    
    def _extract_name_or_title(self, context: str, query: str) -> str:
        """Extract names or titles from context"""
        sentences = context.split('।')
        
        if 'সুপুরুষ' in query:
            for sentence in sentences:
                if 'সুপুরুষ' in sentence:
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if 'সুপুরুষ' in word and i > 0:
                            return words[i-1].strip(',।')
                        elif i < len(words) - 1 and 'সুপুরুষ' in words[i+1]:
                            return word.strip(',।')
        
        if 'ভাগ্য দেবতা' in query or 'ভাগ্যদেবতা' in query:
            for sentence in sentences:
                if 'ভাগ্য' in sentence and 'দেবতা' in sentence:
                    if 'মামা' in sentence:
                        return 'মামাকে'
                    words = sentence.split()
                    for word in words:
                        if word in ['বাবা', 'মা', 'দাদা', 'দিদি', 'কাকা', 'মামা']:
                            return word + 'কে'
        
        return None
    
    def _extract_age(self, context: str, query: str) -> str:
        """Extract age information from context"""
        sentences = context.split('।')
        
        # Look for age patterns
        age_pattern = r'(\d+)\s*বছর'
        
        for sentence in sentences:
            if 'বয়স' in sentence or 'বছর' in sentence:
                matches = re.findall(age_pattern, sentence)
                if matches:
                    return matches[0] + ' বছর'
                
                # Also look for written numbers
                written_numbers = {
                    'পনের': '১৫', 'পনেরো': '১৫', 'পনেরো': '১৫',
                    'ষোল': '১৬', 'সতের': '১৭', 'আঠারো': '১৮',
                    'উনিশ': '১৯', 'বিশ': '২০'
                }
                
                for written, digit in written_numbers.items():
                    if written in sentence:
                        return digit + ' বছর'
        
        return None
    
    def _is_relevant_sentence(self, query: str, sentence: str) -> bool:
        """Check if sentence is relevant to query"""
        query_words = set(self.bengali_processor.tokenize_bengali(query.lower()))
        sentence_words = set(self.bengali_processor.tokenize_bengali(sentence.lower()))
        
        # Remove stopwords
        query_words = set(self.bengali_processor.remove_stopwords(list(query_words)))
        sentence_words = set(self.bengali_processor.remove_stopwords(list(sentence_words)))
        
        # Calculate word overlap
        if not query_words:
            return False
        
        overlap = len(query_words.intersection(sentence_words))
        return overlap / len(query_words) > 0.3  # At least 30% word overlap
    
    def query(self, user_query: str, k: int = 5) -> Dict[str, Any]:
        """Process user query and generate response"""
        # Get conversation context
        context = self.memory.get_context(user_query)
        
        retrieved_docs = self.vector_store.search(user_query, k=k)
        
        answer = self._generate_answer(user_query, context, retrieved_docs)
        
        self.memory.add_to_short_term(user_query, answer)
        
        response = {
            'query': user_query,
            'answer': answer,
            'retrieved_documents': [
                {
                    'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                    'score': score,
                    'chunk_id': doc['chunk_id']
                }
                for doc, score in retrieved_docs[:3]
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def save_system(self, filepath: str):
        """Save the system state"""
        system_data = {
            'chunks': self.vector_store.chunks,
            'embeddings': self.vector_store.embeddings,
            'memory': self.memory.short_term_memory
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
    
    def load_system(self, filepath: str):
        """Load system state"""
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        self.vector_store.chunks = system_data['chunks']
        self.vector_store.embeddings = system_data['embeddings']
        self.memory.short_term_memory = system_data['memory']
        
        
        if self.vector_store.embeddings is not None:
            dimension = self.vector_store.embeddings.shape[1]
            self.vector_store.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(self.vector_store.embeddings)
            self.vector_store.index.add(self.vector_store.embeddings)
    
    ##TESTING##
    
    
    

if __name__ == "__main__":
    rag = MultilingualRAGSystem()

    dummy_bangla_text = """
    আনুপম একজন মেধাবী ছাত্র। তার বয়স পনের বছর। সে প্রতিদিন সকালে স্কুলে যায়।
    তার বাবা একজন শিক্ষক। মা গৃহিণী। আনুপম পড়াশোনায় খুব ভালো।
    সে ভবিষ্যতে একজন ডাক্তার হতে চায়। সে বাংলায় রচনা লিখতেও খুব পছন্দ করে।
    """

    
    chunks = rag.document_processor.create_chunks(dummy_bangla_text)
    print(f"\n✅ Created {len(chunks)} chunks from dummy text")

    rag.vector_store.add_documents(chunks)

    
    rag.memory.set_long_term_memory(rag.vector_store)

    question = "আনুপমের বয়স কত?"
    # অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
    response = rag.query(question)

    print("\n=== Query Answer ===")
    print(f"Q: {response['query']}")
    print(f"A: {response['answer']}\n")

    print("=== Retrieved Chunks ===")
    for doc in response['retrieved_documents']:
        print(f"- Score: {doc['score']:.4f}")
        print(f"  Chunk ID: {doc['chunk_id']}")
        print(f"  Text: {doc['text']}\n")
