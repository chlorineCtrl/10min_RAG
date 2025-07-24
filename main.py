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
import torch
from dotenv import load_dotenv

load_dotenv()

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

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
    
    def get_short_term_memory(self) -> List[Dict]:
        return self.short_term_memory


class MultilingualRAGSystem:
    """Main RAG system combining all components with OpenAI GPT-4.1"""
    
    def __init__(self, github_token: str = None):
        self.document_processor = DocumentProcessor()
        self.vector_store = MultilingualVectorStore()
        self.memory = ConversationMemory()
        self.bengali_processor = BengaliTextProcessor()
        
        # Initialize OpenAI GPT-4.1 client
        self.endpoint = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1"
        self.token = github_token or os.environ.get("GITHUB_TOKEN")
        
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable or pass it to constructor.")
        
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        self.system_prompt = """You are an intelligent multilingual assistant specializing in Bengali and English text analysis. Your task is to answer questions based on provided context from documents.

Key instructions:
1. Answer questions accurately based on the provided context
2. If the context contains the answer, provide a direct and concise response
3. For Bengali questions, respond in Bengali; for English questions, respond in English
4. If you cannot find the answer in the provided context, say so clearly
5. Be precise and avoid making up information not present in the context
6. When extracting specific information like names, ages, or titles, quote directly from the context
7. Maintain the original language and tone of the source material when possible

Context format: You will receive relevant document chunks and conversation history to help answer the current question."""

    def load_knowledge_base(self, pdf_path: str):
        """Load and process PDF document into knowledge base"""
        print(f"Loading knowledge base from: {pdf_path}")
        
        raw_text = self.document_processor.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            raise ValueError("Could not extract text from PDF")
        
        chunks = self.document_processor.create_chunks(raw_text)
        print(f"Created {len(chunks)} chunks from document")
        
        self.vector_store.add_documents(chunks)
        self.memory.set_long_term_memory(self.vector_store)
        
        return len(chunks)
    
    def _generate_answer_with_gpt4(self, query: str, context: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Generate answer using OpenAI GPT-4.1"""
        try:
            relevant_context = ""
            for i, (doc, score) in enumerate(retrieved_docs[:3]):
                relevant_context += f"Context {i+1} (Relevance: {score:.3f}):\n{doc['text']}\n\n"
            
            user_message = f"""Based on the following context, please answer this question:

Question: {query}

Available Context:
{relevant_context}

Previous Conversation Context:
{context}

Please provide a direct and accurate answer based on the available context."""

            response = self.client.complete(
                messages=[
                    SystemMessage(self.system_prompt),
                    UserMessage(user_message),
                ],
                temperature=0.3,  
                top_p=0.9,
                max_tokens=500,
                model=self.model
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer with GPT-4.1: {e}")
            return "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে। / Sorry, there was an issue generating the answer."
    
    def query(self, user_query: str, k: int = 5) -> Dict[str, Any]:
        """Process user query and generate response using GPT-4.1"""
        context = self.memory.get_context(user_query)
        
        retrieved_docs = self.vector_store.search(user_query, k=k)
        
        answer = self._generate_answer_with_gpt4(user_query, context, retrieved_docs)
        
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



if __name__ == "__main__":

    rag = MultilingualRAGSystem()

    # dummy 
    dummy_bangla_text = """
    আনুপম একজন মেধাবী ছাত্র। তার বয়স পনের বছর। সে প্রতিদিন সকালে স্কুলে যায়।
    তার বাবা একজন শিক্ষক। মা গৃহিণী। আনুপম পড়াশোনায় খুব ভালো।
    সে ভবিষ্যতে একজন ডাক্তার হতে চায়। সে বাংলায় রচনা লিখতেও খুব পছন্দ করে।
    আনুপমের মামা তাকে সুপুরুষ বলে ডাকেন। পরিবারে সবাই তাকে ভাগ্য দেবতা মনে করে।
    """

    chunks = rag.document_processor.create_chunks(dummy_bangla_text)
    print(f"\n✅ Created {len(chunks)} chunks from dummy text")

    rag.vector_store.add_documents(chunks)
    rag.memory.set_long_term_memory(rag.vector_store)

   
    questions = [
        "আনুপমের বাবার বয়স কত?",
        
    ]

    print("\n=== Testing Questions ===")
    for question in questions:
        response = rag.query(question)
        print(f"\nQ: {response['query']}")
        print(f"A: {response['answer']}")
        print("-" * 50)

    rag.save_system("rag_system_gpt4_state.pkl")
    print("\n✅ Saved system state with GPT-4.1 integration")
    
    print("\n=== Conversation History ===")
    for item in rag.memory.short_term_memory:
        print(f"Q: {item['query']}")
        print(f"A: {item['response']}\n")