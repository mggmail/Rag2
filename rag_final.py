import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import random
from enum import Enum

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
import langchain

# Security & Configuration

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Database

import sqlite3
from contextlib import contextmanager
from collections import defaultdict

# ===========================

# 1. KONFIGURACJA Z A/B TESTING

# ===========================

load_dotenv()

class ConfigVariant(str, Enum):
“”“Warianty konfiguracji dla A/B testów”””
CONTROL = “control”
VARIANT_A = “variant_a”
VARIANT_B = “variant_b”
VARIANT_C = “variant_c”

class RAGConfig(BaseModel):
“”“Bezpieczna konfiguracja z walidacją”””
openai_api_key: str = Field(…, min_length=20)
chunk_size: int = Field(default=500, ge=100, le=2000)
chunk_overlap: int = Field(default=50, ge=0, le=500)
max_tokens: int = Field(default=1000, ge=100, le=4000)
temperature: float = Field(default=0.0, ge=0.0, le=1.0)
top_k: int = Field(default=5, ge=1, le=20)
collection_name: str = Field(default=“rag_collection”)
persist_directory: str = Field(default=”./chroma_db”)
variant: str = Field(default=“control”)
search_type: str = Field(default=“mmr”)  # mmr, similarity, similarity_score_threshold
lambda_mult: float = Field(default=0.7, ge=0.0, le=1.0)

```
@validator('chunk_overlap')
def validate_overlap(cls, v, values):
    if 'chunk_size' in values and v >= values['chunk_size']:
        raise ValueError("chunk_overlap must be less than chunk_size")
    return v
```

class ABTestingManager:
“”“Zarządzanie testami A/B różnych konfiguracji”””

```
def __init__(self, db_path: str = "./feedback.db"):
    self.db_path = db_path
    self.variants = self._define_variants()
    self.init_ab_tables()

def _define_variants(self) -> Dict[str, RAGConfig]:
    """Definicja wariantów konfiguracji do testowania"""
    base_api_key = os.getenv("OPENAI_API_KEY")
    
    return {
        ConfigVariant.CONTROL: RAGConfig(
            openai_api_key=base_api_key,
            chunk_size=500,
            chunk_overlap=50,
            top_k=5,
            temperature=0.0,
            variant="control",
            search_type="mmr",
            lambda_mult=0.7
        ),
        ConfigVariant.VARIANT_A: RAGConfig(
            openai_api_key=base_api_key,
            chunk_size=800,
            chunk_overlap=100,
            top_k=7,
            temperature=0.1,
            variant="variant_a",
            search_type="mmr",
            lambda_mult=0.5
        ),
        ConfigVariant.VARIANT_B: RAGConfig(
            openai_api_key=base_api_key,
            chunk_size=300,
            chunk_overlap=30,
            top_k=10,
            temperature=0.0,
            variant="variant_b",
            search_type="similarity",
            lambda_mult=0.7
        ),
        ConfigVariant.VARIANT_C: RAGConfig(
            openai_api_key=base_api_key,
            chunk_size=600,
            chunk_overlap=80,
            top_k=5,
            temperature=0.2,
            variant="variant_c",
            search_type="mmr",
            lambda_mult=0.8
        )
    }

@contextmanager
def get_connection(self):
    """Context manager dla połączeń DB"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_ab_tables(self):
    """Inicjalizacja tabel dla A/B testów"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela wariantów
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_name TEXT UNIQUE NOT NULL,
                config_json TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela przypisań użytkowników
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela wyników A/B
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_name TEXT NOT NULL,
                total_queries INTEGER DEFAULT 0,
                avg_rating REAL,
                avg_response_time REAL,
                avg_cost REAL,
                success_rate REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def assign_variant(self, session_id: str) -> str:
    """Przypisanie wariantu do sesji (równomierny rozkład)"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Sprawdź czy sesja już ma wariant
        cursor.execute('''
            SELECT variant_name FROM ab_test_assignments
            WHERE session_id = ?
            ORDER BY assigned_at DESC LIMIT 1
        ''', (session_id,))
        
        result = cursor.fetchone()
        if result:
            return result['variant_name']
        
        # Przypisz nowy wariant (równy rozkład)
        active_variants = list(self.variants.keys())
        variant = random.choice(active_variants)
        
        cursor.execute('''
            INSERT INTO ab_test_assignments (session_id, variant_name)
            VALUES (?, ?)
        ''', (session_id, variant))
        
        conn.commit()
        
        return variant

def get_variant_config(self, variant_name: str) -> RAGConfig:
    """Pobranie konfiguracji wariantu"""
    return self.variants.get(variant_name, self.variants[ConfigVariant.CONTROL])

def update_variant_stats(self):
    """Aktualizacja statystyk wariantów"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        for variant_name in self.variants.keys():
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(user_rating) as avg_rating,
                    AVG(response_time) as avg_time,
                    AVG(cost) as avg_cost,
                    SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM query_feedback qf
                JOIN ab_test_assignments ata ON qf.session_id = ata.session_id
                WHERE ata.variant_name = ?
            ''', (variant_name,))
            
            stats = cursor.fetchone()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ab_test_results 
                (variant_name, total_queries, avg_rating, avg_response_time, 
                 avg_cost, success_rate, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (variant_name, stats['total'], stats['avg_rating'],
                  stats['avg_time'], stats['avg_cost'], stats['success_rate']))
        
        conn.commit()

def get_variant_comparison(self) -> Dict[str, Any]:
    """Porównanie wyników wszystkich wariantów"""
    self.update_variant_stats()
    
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM ab_test_results
            WHERE total_queries > 0
            ORDER BY avg_rating DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'variant': row['variant_name'],
                'queries': row['total_queries'],
                'avg_rating': round(row['avg_rating'] or 0, 2),
                'avg_response_time': round(row['avg_response_time'] or 0, 2),
                'avg_cost': round(row['avg_cost'] or 0, 6),
                'success_rate': round(row['success_rate'] or 0, 2)
            })
        
        # Znajdź zwycięzcę
        winner = max(results, key=lambda x: x['avg_rating']) if results else class EnhancedFeedbackSystem:
"""Rozszerzony system feedbacku z oceną użytkowników"""

def __init__(self, db_path: str = "./feedback.db"):
    self.db_path = db_path
    self.init_database()
    self.logger = logging.getLogger(__name__)

@contextmanager
def get_connection(self):
    """Context manager dla połączeń DB"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database(self):
    """Inicjalizacja rozszerzonej bazy danych"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela głównych zapytań
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                variant_name TEXT,
                query TEXT NOT NULL,
                response TEXT,
                user_rating INTEGER CHECK(user_rating >= 1 AND user_rating <= 5),
                feedback_comment TEXT,
                retrieved_docs TEXT,
                num_docs_retrieved INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                tokens_used INTEGER,
                cost REAL,
                model_used TEXT,
                is_successful INTEGER DEFAULT 1,
                retrieval_score REAL
            )
        ''')
        
        return {
            'variants': results,
            'winner': winner,
            'recommendation': self._generate_recommendation(results)
        }

def _generate_recommendation(self, results: List[Dict]) -> str:
    """Generowanie rekomendacji na podstawie wyników"""
    if not results or len(results) < 2:
        return "Zbyt mało danych do porównania"
    
    winner = max(results, key=lambda x: x['avg_rating'])
    
    if winner['avg_rating'] > 4.0:
        return f"Wariant '{winner['variant']}' znacząco przewyższa inne - rozważ wdrożenie jako domyślny"
    elif winner['avg_rating'] > 3.5:
        return f"Wariant '{winner['variant']}' pokazuje obiecujące wyniki - kontynuuj testy"
    else:
        return "Wszystkie warianty wymagają optymalizacji - rozważ nowe konfiguracje"
```

class SecurityManager:
“”“Zarządzanie bezpieczeństwem i sanityzacją”””

```
@staticmethod
def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Oczyszczanie inputu użytkownika"""
    if not text or len(text.strip()) == 0:
        raise ValueError("Input cannot be empty")
    
    text = text[:max_length]
    dangerous_patterns = ['<script>', 'javascript:', 'onerror=']
    for pattern in dangerous_patterns:
        text = text.replace(pattern, '')
    
    return text.strip()

@staticmethod
def validate_api_key(api_key: str) -> bool:
    """Walidacja klucza API"""
    return api_key.startswith(('sk-', 'sk-proj-')) and len(api_key) > 20
```

# ===========================

# 2. AUTOMATYCZNE RE-INDEXOWANIE

# ===========================

class AutoReindexingManager:
“”“System automatycznego re-indexowania na podstawie feedbacku”””

```
def __init__(self, db_path: str = "./feedback.db"):
    self.db_path = db_path
    self.logger = logging.getLogger(__name__)
    self.init_reindex_tables()

@contextmanager
def get_connection(self):
    """Context manager dla połączeń DB"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_reindex_tables(self):
    """Inicjalizacja tabel dla re-indexowania"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela dokumentów do re-indexowania
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reindex_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_path TEXT NOT NULL,
                reason TEXT,
                priority INTEGER DEFAULT 1,
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed_at DATETIME
            )
        ''')
        
        # Tabela historii re-indexowania
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reindex_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_reason TEXT,
                documents_reindexed INTEGER,
                old_chunk_size INTEGER,
                new_chunk_size INTEGER,
                performance_before REAL,
                performance_after REAL,
                executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela dokumentów problematycznych
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problematic_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT UNIQUE,
                document_source TEXT,
                low_rating_count INTEGER DEFAULT 0,
                avg_rating REAL,
                retrieval_frequency INTEGER DEFAULT 0,
                last_issue_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def analyze_feedback_for_reindexing(self) -> Dict[str, Any]:
    """Analiza feedbacku do określenia potrzeby re-indexowania"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Znajdź dokumenty z niską oceną
        cursor.execute('''
            SELECT 
                sd.doc_metadata,
                COUNT(*) as usage_count,
                AVG(qf.user_rating) as avg_rating
            FROM source_documents sd
            JOIN query_feedback qf ON sd.query_id = qf.id
            WHERE qf.user_rating IS NOT NULL AND qf.user_rating <= 2
            GROUP BY sd.doc_metadata
            HAVING usage_count >= 3
            ORDER BY avg_rating ASC, usage_count DESC
        ''')
        
        problematic_docs = []
        for row in cursor.fetchall():
            try:
                metadata = json.loads(row['doc_metadata'])
                problematic_docs.append({
                    'source': metadata.get('source', 'Unknown'),
                    'doc_hash': metadata.get('doc_hash', 'Unknown'),
                    'usage_count': row['usage_count'],
                    'avg_rating': round(row['avg_rating'], 2)
                })
            except:
                pass
        
        # Sprawdź ogólną wydajność
        cursor.execute('''
            SELECT 
                AVG(user_rating) as avg_rating,
                AVG(response_time) as avg_time
            FROM query_feedback
            WHERE timestamp >= datetime('now', '-7 days')
            AND user_rating IS NOT NULL
        ''')
        
        overall = cursor.fetchone()
        
        needs_reindex = False
        reason = []
        
        if overall['avg_rating'] and overall['avg_rating'] < 3.0:
            needs_reindex = True
            reason.append("Ogólna ocena poniżej 3.0")
        
        if len(problematic_docs) >= 5:
            needs_reindex = True
            reason.append(f"Znaleziono {len(problematic_docs)} problematycznych dokumentów")
        
        if overall['avg_time'] and overall['avg_time'] > 5.0:
            needs_reindex = True
            reason.append("Długi czas odpowiedzi (>5s)")
        
        return {
            'needs_reindex': needs_reindex,
            'reasons': reason,
            'problematic_documents': problematic_docs,
            'overall_rating': round(overall['avg_rating'] or 0, 2),
            'overall_time': round(overall['avg_time'] or 0, 2)
        }

def queue_reindexing(self, document_paths: List[str], reason: str, priority: int = 1):
    """Dodanie dokumentów do kolejki re-indexowania"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        for path in document_paths:
            cursor.execute('''
                INSERT INTO reindex_queue (document_path, reason, priority)
                VALUES (?, ?, ?)
            ''', (path, reason, priority))
        
        conn.commit()
        
        self.logger.info(f"Queued {len(document_paths)} documents for reindexing: {reason}")

def auto_trigger_reindexing(self, rag_system, threshold_days: int = 7) -> bool:
    """Automatyczne uruchamianie re-indexowania"""
    analysis = self.analyze_feedback_for_reindexing()
    
    if not analysis['needs_reindex']:
        self.logger.info("No reindexing needed at this time")
        return False
    
    self.logger.warning(f"Reindexing triggered: {', '.join(analysis['reasons'])}")
    
    # Zapisz do historii
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reindex_history 
            (trigger_reason, performance_before)
            VALUES (?, ?)
        ''', (', '.join(analysis['reasons']), analysis['overall_rating']))
        conn.commit()
    
    # Wykonaj re-indexowanie z optymalnymi parametrami
    try:
        self._execute_smart_reindexing(rag_system, analysis)
        return True
    except Exception as e:
        self.logger.error(f"Reindexing failed: {e}")
        return False

def _execute_smart_reindexing(self, rag_system, analysis: Dict):
    """Inteligentne re-indexowanie z dostosowanymi parametrami"""
    # Analiza optymalnych parametrów
    optimal_params = self._calculate_optimal_params(analysis)
    
    self.logger.info(f"Reindexing with params: {optimal_params}")
    
    # Backup starej konfiguracji
    old_chunk_size = rag_system.config.chunk_size
    
    # Aktualizacja konfiguracji
    rag_system.config.chunk_size = optimal_params['chunk_size']
    rag_system.config.chunk_overlap = optimal_params['chunk_overlap']
    
    # Re-indexowanie dokumentów
    source_path = "./knowledge_base"
    if os.path.exists(source_path):
        texts = rag_system.load_and_process_documents(source_path)
        
        # Usuń stary vector store
        import shutil
        if os.path.exists(rag_system.config.persist_directory):
            shutil.rmtree(rag_system.config.persist_directory)
        
        # Stwórz nowy
        rag_system.create_vectorstore(texts)
        rag_system.setup_retriever()
        rag_system.create_qa_chain()
        
        # Zapisz do historii
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE reindex_history 
                SET documents_reindexed = ?,
                    old_chunk_size = ?,
                    new_chunk_size = ?
                WHERE id = (SELECT MAX(id) FROM reindex_history)
            ''', (len(texts), old_chunk_size, optimal_params['chunk_size']))
            conn.commit()
        
        self.logger.info(f"Reindexing completed: {len(texts)} chunks created")

def _calculate_optimal_params(self, analysis: Dict) -> Dict[str, int]:
    """Kalkulacja optymalnych parametrów na podstawie analizy"""
    # Domyślne wartości
    chunk_size = 500
    chunk_overlap = 50
    
    # Jeśli długi czas odpowiedzi - zmniejsz chunk size
    if analysis['overall_time'] > 5.0:
        chunk_size = 300
        chunk_overlap = 30
    
    # Jeśli niska ocena - zwiększ chunk size dla lepszego kontekstu
    elif analysis['overall_rating'] < 3.0:
        chunk_size = 700
        chunk_overlap = 70
    
    # Jeśli wiele problematycznych dokumentów - średnie chunki
    elif len(analysis['problematic_documents']) > 5:
        chunk_size = 450
        chunk_overlap = 50
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }

def get_reindexing_stats(self) -> Dict[str, Any]:
    """Statystyki re-indexowania"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_reindexes,
                AVG(performance_after - performance_before) as avg_improvement
            FROM reindex_history
            WHERE performance_after IS NOT NULL
        ''')
        
        stats = cursor.fetchone()
        
        cursor.execute('''
            SELECT * FROM reindex_history
            ORDER BY executed_at DESC
            LIMIT 5
        ''')
        
        recent = [dict(row) for row in cursor.fetchall()]
        
        return {
            'total_reindexes': stats['total_reindexes'] or 0,
            'avg_improvement': round(stats['avg_improvement'] or 0, 2),
            'recent_reindexes': recent
        }
```

class EnhancedFeedbackSystem:
“”“Rozszerzony system feedbacku z oceną użytkowników”””

```
def __init__(self, db_path: str = "./feedback.db"):
    self.db_path = db_path
    self.init_database()
    self.logger = logging.getLogger(__name__)

@contextmanager
def get_connection(self):
    """Context manager dla połączeń DB"""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database(self):
    """Inicjalizacja rozszerzonej bazy danych"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela głównych zapytań
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT NOT NULL,
                response TEXT,
                user_rating INTEGER CHECK(user_rating >= 1 AND user_rating <= 5),
                feedback_comment TEXT,
                retrieved_docs TEXT,
                num_docs_retrieved INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                tokens_used INTEGER,
                cost REAL,
                model_used TEXT,
                is_successful INTEGER DEFAULT 1,
                retrieval_score REAL
            )
        ''')
        
        # Tabela nieudanych zapytań
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                attempted_fixes TEXT
            )
        ''')
        
        # Tabela dokumentów źródłowych
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                doc_content TEXT,
                doc_metadata TEXT,
                relevance_score REAL,
                was_helpful INTEGER,
                FOREIGN KEY (query_id) REFERENCES query_feedback(id)
            )
        ''')
        
        # Tabela trendów i wzorców
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_value TEXT,
                frequency INTEGER DEFAULT 1,
                avg_rating REAL,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela ulepszeń
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                improvement_type TEXT,
                description TEXT,
                before_metric REAL,
                after_metric REAL,
                implemented_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Indeksy dla wydajności
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_query_rating 
            ON query_feedback(user_rating)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_query_timestamp 
            ON query_feedback(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_failed_timestamp 
            ON failed_queries(timestamp)
        ''')
        
        conn.commit()

def log_query(self, session_id: str, query: str, response: str, 
               docs: List, response_time: float, tokens_used: int,
               cost: float, model_used: str, variant_name: str = # ===========================
```

# 6. PRZYKŁADY UŻYCIA Z A/B TESTING I AUTO-REINDEXING

# ===========================

def interactive_demo_with_ab():
“”“Interaktywna demonstracja z A/B testing”””

```
print("=" * 60)
print("ADVANCED RAG SYSTEM - A/B TESTING MODE")
print("=" * 60)

# Włączenie A/B testing
rag_system = AdvancedRAGSystem(enable_ab_testing=True)

print(f"\n🧪 Twoja sesja używa wariantu: {rag_system.config.variant}")
print(f"   Chunk size: {rag_system.config.chunk_size}")
print(f"   Top K: {rag_system.config.top_k}")
print(f"   Search type: {rag_system.config.search_type}")

# Ładowanie danych
knowledge_base_path = "./knowledge_base"
persist_dir = f"{rag_system.config.persist_directory}_{rag_system.config.variant}"

if not os.path.exists(persist_dir):
    print("\n📚 Ładowanie dokumentów...")
    texts = rag_system.load_and_process_documents(knowledge_base_path)
    rag_system.create_vectorstore(texts)
else:
    print("\n📚 Ładowanie istniejącej bazy...")
    rag_system.load_vectorstore()

rag_system.setup_retriever()
rag_system.create_qa_chain()

print("✅ System gotowy!\n")

# Pętla interaktywna
while True:
    print("\n" + "=" * 60)
    question = input("❓ Pytanie ('stats'/'ab'/'reindex'/'quit'): ").strip()
    
    if question.lower() == 'quit':
        break
    
    if question.lower() == 'stats':
        stats = rag_system.get_system_stats()
        print("\n📊 STATYSTYKI:")
        print(f"Zapytania: {stats['total_queries']}")
        print(f"Średnia ocena: {stats['avg_rating']}/5.0")
        print(f"Czas odpowiedzi: {stats['avg_response_time']:.2f}s")
        continue
    
    if question.lower() == 'ab':
        results = rag_system.get_ab_test_results()
        print("\n🧪 WYNIKI A/B TESTING:")
        print("=" * 60)
        
        for variant in results['variants']:
            print(f"\n📊 Wariant: {variant['variant']}")
            print(f"   Zapytania: {variant['queries']}")
            print(f"   Ocena: {variant['avg_rating']}/5.0")
            print(f"   Czas: {variant['avg_response_time']:.2f}s")
            print(f"   Koszt: ${variant['avg_cost']:.6f}")
            print(f"   Sukces: {variant['success_rate']:.1f}%")
        
        if results['winner']:
            print(f"\n🏆 ZWYCIĘZCA: {results['winner']['variant']}")
            print(f"   Ocena: {results['winner']['avg_rating']}/5.0")
        
        print(f"\n💡 Rekomendacja:")
        print(f"   {results['recommendation']}")
        continue
    
    if question.lower() == 'reindex':
        print("\n🔄 Sprawdzanie potrzeby re-indexowania...")
        analysis = rag_system.check_and_trigger_reindexing()
        
        print("\n📊 ANALIZA:")
        print(f"Średnia ocena: {analysis['overall_rating']}/5.0")
        print(f"Średni czas: {analysis['overall_time']:.2f}s")
        print(f"Problematyczne dokumenty: {len(analysis['problematic_documents'])}")
        
        if analysis['needs_reindex']:
            print("\n⚠️ RE-INDEXING POTRZEBNY!")
            print("Powody:")
            for reason in analysis['reasons']:
                print(f"  • {reason}")
            
            if analysis['reindex_executed']:
                print("\n✅ Re-indexing wykonany pomyślnie!")
            else:
                print("\n❌ Re-indexing nie powiódł się")
        else:
            print("\n✅ System działa optymalnie - re-indexing niepotrzebny")
        
        continue
    
    if not question:
        continue
    
    # Zapytanie
    try:
        print("\n🤔 Przetwarzam...")
        result = rag_system.query(question)
        
        print("\n" + "=" * 60)
        print("💬 ODPOWIEDŹ:")
        print("=" * 60)
        print(result['answer'])
        
        print(f"\n🧪 Wariant: {result['variant']}")
        print(f"⚡ Czas: {result['metrics']['response_time']:.2f}s")
        print(f"💰 Koszt: ${result['metrics']['total_cost']:.4f}")
        
        # Ocena
        while True:
            rating_input = input("\n⭐ Oceń (1-5, enter=pomiń): ").strip()
            
            if not rating_input:
                break
            
            try:
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    comment = input("Komentarz (opcjonalnie): ").strip()
                    rag_system.rate_response(
                        result['query_id'], 
                        rating, 
                        comment if comment else None
                    )
                    print(f"✅ Zapisano ocenę: {rating} ⭐")
                    break
                else:
                    print("❌ Ocena 1-5")
            except ValueError:
                print("❌ Wpisz liczbę 1-5")
    
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")

print("\n👋 Dziękujemy!")
```

def ab_testing_simulation():
“”“Symulacja testów A/B z wieloma użytkownikami”””

```
print("=" * 60)
print("A/B TESTING SIMULATION")
print("=" * 60)

test_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does AI work?",
    "What is deep learning?",
    "Describe natural language processing"
]

# Symulacja 20 sesji (5 na wariant)
num_sessions = 20

print(f"\n🧪 Symulacja {num_sessions} sesji użytkowników...\n")

for session_num in range(num_sessions):
    # Każda sesja to nowy system
    rag_system = AdvancedRAGSystem(enable_ab_testing=True)
    
    variant = rag_system.config.variant
    print(f"Sesja {session_num + 1}/20: Wariant {variant}")
    
    # Ładowanie/tworzenie vectorstore
    knowledge_base_path = "./knowledge_base"
    persist_dir = f"{rag_system.config.persist_directory}_{variant}"
    
    if not os.path.exists(persist_dir):
        texts = rag_system.load_and_process_documents(knowledge_base_path)
        rag_system.create_vectorstore(texts)
    else:
        rag_system.load_vectorstore()
    
    rag_system.setup_retriever()
    rag_system.create_qa_chain()
    
    # Losowe zapytanie
    query = random.choice(test_queries)
    
    try:
        result = rag_system.query(query)
        
        # Symulowana ocena (losowa z tendencją)
        # Control i Variant A dostają wyższe oceny
        if variant in ['control', 'variant_a']:
            rating = random.choices([3, 4, 5], weights=[1, 3, 6])[0]
        else:
            rating = random.choices([2, 3, 4, 5], weights=[1, 3, 4, 2])[0]
        
        rag_system.rate_response(result['query_id'], rating)
        
        print(f"  ✅ Zapytanie: '{query[:40]}...'")
        print(f"  ⭐ Ocena: {rating}/5, Czas: {result['metrics']['response_time']:.2f}s")
    
    except Exception as e:
        print(f"  ❌ Błąd: {e}")

# Analiza wyników
print("\n" + "=" * 60)
print("📊 WYNIKI TESTÓW A/B")
print("=" * 60)

ab_manager = ABTestingManager()
results = ab_manager.get_variant_comparison()

print("\nPorównanie wariantów:\n")

for variant in sorted(results['variants'], key=lambda x: x['avg_rating'], reverse=True):
    stars = "⭐" * int(variant['avg_rating'])
    print(f"{variant['variant']:12} | {stars} {variant['avg_rating']:.2f}/5.0 | "
          f"Zapytania: {variant['queries']:2} | "
          f"Czas: {variant['avg_response_time']:.2f}s")

if results['winner']:
    print(f"\n🏆 ZWYCIĘZCA: {results['winner']['variant']}")
    print(f"   Konfiguracja:")
    winner_config = ab_manager.get_variant_config(results['winner']['variant'])
    print(f"   - Chunk size: {winner_config.chunk_size}")
    print(f"   - Chunk overlap: {winner_config.chunk_overlap}")
    print(f"   - Top K: {winner_config.top_k}")
    print(f"   - Search type: {winner_config.search_type}")

print(f"\n💡 {results['recommendation']}")
```

def auto_reindexing_demo():
“”“Demonstracja automatycznego re-indexowania”””

```
print("=" * 60)
print("AUTO-REINDEXING DEMO")
print("=" * 60)

config = RAGConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)

rag_system = AdvancedRAGSystem(config=config)

# Ładowanie danych
knowledge_base_path = "./knowledge_base"

if not os.path.exists(rag_system.config.persist_directory):
    print("\n📚 Inicjalizacja systemu...")
    texts = rag_system.load_and_process_documents(knowledge_base_path)
    rag_system.create_vectorstore(texts)
else:
    print("\n📚 Ładowanie...")
    rag_system.load_vectorstore()

rag_system.setup_retriever()
rag_system.create_qa_chain()

# Symulacja złych wyników
print("\n🔄 Symulacja zapytań z niskimi ocenami...\n")

bad_queries = [
    "Tell me something random",
    "What's the weather?",
    "Random question here",
    "Another bad query",
    "This won't work well"
]

for query in bad_queries:
    try:
        result = rag_system.query(query)
        # Niskie oceny
        rag_system.rate_response(result['query_id'], random.choice([1, 2]))
        print(f"✅ Zapytanie: '{query}' - ocena: 1-2⭐")
    except:
        print(f"❌ Błąd: '{query}'")

# Analiza
print("\n" + "=" * 60)
print("📊 ANALIZA SYSTEMU")
print("=" * 60)

analysis = rag_system.check_and_trigger_reindexing()

print(f"\nOgólna ocena: {analysis['overall_rating']}/5.0")
print(f"Średni czas: {analysis['overall_time']:.2f}s")
print(f"Problematyczne dokumenty: {len(analysis['problematic_documents'])}")

if analysis['needs_reindex']:
    print("\n⚠️ SYSTEM WYMAGA RE-INDEXOWANIA")
    print("\nPowody:")
    for reason in analysis['reasons']:
        print(f"  🔴 {reason}")
    
    if analysis.get('reindex_executed'):
        print("\n✅ Auto-reindexing wykonany!")
        
        # Statystyki re-indexowania
        reindex_stats = rag_system.reindex_manager.get_reindexing_stats()
        print(f"\nStatystyki re-indexowania:")
        print(f"  Łącznie wykonanych: {reindex_stats['total_reindexes']}")
        print(f"  Średnia poprawa: {reindex_stats['avg_improvement']:.2f}")
    else:
        print("\n⚠️ Auto-reindexing nie został uruchomiony")
else:
    print("\n✅ System działa optymalnie")

# Sugestie
print("\n" + "=" * 60)
print("💡 SUGESTIE ULEPSZEŃ")
print("=" * 60)

suggestions = rag_system.get_improvement_suggestions()

print(f"\nStan: {suggestions['overall_health'].upper()}")

if suggestions['suggestions']:
    print("\nZalecane działania:")
    for sug in suggestions['suggestions']:
        priority_icon = "🔴" if sug['priority'] == 'high' else "🟡"
        print(f"\n{priority_icon} [{sug['type'].upper()}]")
        print(f"   {sug['suggestion']}")
```

# ===========================

# 7. MAIN

# ===========================

if **name** == “**main**”:
print(”\n🚀 ADVANCED RAG SYSTEM”)
print(”\nWybierz tryb:”)
print(“1. Interaktywny z A/B testing”)
print(“2. Symulacja A/B testing (20 sesji)”)
print(“3. Demo auto-reindexing”)
print(“4. Dashboard analityczny”)
print(“5. Wyjście\n”)

```
choice = input("Wybór (1-5): ").strip()

if choice == "1":
    interactive_demo_with_ab()
elif choice == "2":
    ab_testing_simulation()
elif choice == "3":
    auto_reindexing_demo()
elif choice == "4":
    generate_analytics_dashboard()
else:
    print("👋 Do zobaczenia!"),
               retrieval_score: float = None) -> int:
    """Logowanie zapytania z pełnymi detalami i wariantem A/B"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        docs_summary = json.dumps([
            {
                'content': d.page_content[:300],
                'source': d.metadata.get('source', 'Unknown')
            }
            for d in docs
        ])
        
        cursor.execute('''
            INSERT INTO query_feedback 
            (session_id, variant_name, query, response, retrieved_docs, num_docs_retrieved,
             response_time, tokens_used, cost, model_used, retrieval_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, variant_name, query, response, docs_summary, len(docs),
              response_time, tokens_used, cost, model_used, retrieval_score))
        
        query_id = cursor.lastrowid
        
        # Zapisywanie dokumentów źródłowych
        for doc in docs:
            cursor.execute('''
                INSERT INTO source_documents 
                (query_id, doc_content, doc_metadata)
                VALUES (?, ?, ?)
            ''', (query_id, doc.page_content, json.dumps(doc.metadata)))
        
        conn.commit()
        
        self.logger.info(f"Query logged with ID: {query_id}, variant: {variant_name}")
        return query_id

def add_user_rating(self, query_id: int, rating: int, 
                    comment: str = None) -> bool:
    """Dodawanie oceny użytkownika z walidacją"""
    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5")
    
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE query_feedback 
            SET user_rating = ?, feedback_comment = ?
            WHERE id = ?
        ''', (rating, comment, query_id))
        
        if cursor.rowcount == 0:
            self.logger.warning(f"Query ID {query_id} not found")
            return False
        
        conn.commit()
        
        self.logger.info(f"Rating {rating} added for query {query_id}")
        return True

def log_error(self, session_id: str, query: str, error_type: str,
              error_message: str, stack_trace: str = None):
    """Logowanie błędów z kategoryzacją"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO failed_queries 
            (session_id, query, error_type, error_message, stack_trace)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, query, error_type, error_message, stack_trace))
        
        conn.commit()
        
        self.logger.error(f"Error logged: {error_type} - {error_message}")

def get_analytics(self, days: int = 30) -> Dict[str, Any]:
    """Szczegółowa analityka systemu"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        date_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Podstawowe statystyki
        cursor.execute('''
            SELECT 
                COUNT(*) as total_queries,
                AVG(response_time) as avg_response_time,
                AVG(user_rating) as avg_rating,
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as positive_ratings,
                COUNT(CASE WHEN user_rating <= 2 THEN 1 END) as negative_ratings
            FROM query_feedback
            WHERE timestamp >= ?
        ''', (date_threshold,))
        
        stats = cursor.fetchone()
        
        # Statystyki błędów
        cursor.execute('''
            SELECT 
                COUNT(*) as total_errors,
                COUNT(DISTINCT error_type) as unique_error_types
            FROM failed_queries
            WHERE timestamp >= ?
        ''', (date_threshold,))
        
        errors = cursor.fetchone()
        
        # Top błędy
        cursor.execute('''
            SELECT error_type, COUNT(*) as count
            FROM failed_queries
            WHERE timestamp >= ?
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 5
        ''', (date_threshold,))
        
        top_errors = [
            {'error_type': row['error_type'], 'count': row['count']}
            for row in cursor.fetchall()
        ]
        
        # Rozkład ocen
        cursor.execute('''
            SELECT user_rating, COUNT(*) as count
            FROM query_feedback
            WHERE user_rating IS NOT NULL AND timestamp >= ?
            GROUP BY user_rating
            ORDER BY user_rating
        ''', (date_threshold,))
        
        rating_distribution = {
            row['user_rating']: row['count']
            for row in cursor.fetchall()
        }
        
        # Success rate
        total = stats['total_queries']
        total_errors_count = errors['total_errors']
        success_rate = (total / (total + total_errors_count) * 100) if (total + total_errors_count) > 0 else 0
        
        return {
            'period_days': days,
            'total_queries': total,
            'successful_queries': total,
            'failed_queries': total_errors_count,
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(stats['avg_response_time'] or 0, 2),
            'avg_rating': round(stats['avg_rating'] or 0, 2),
            'total_tokens': stats['total_tokens'] or 0,
            'total_cost': round(stats['total_cost'] or 0, 4),
            'positive_ratings': stats['positive_ratings'] or 0,
            'negative_ratings': stats['negative_ratings'] or 0,
            'unique_error_types': errors['unique_error_types'] or 0,
            'rating_distribution': rating_distribution,
            'top_errors': top_errors
        }

def get_low_rated_queries(self, threshold: int = 2, limit: int = 10) -> List[Dict]:
    """Zapytania z niską oceną do analizy"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, query, response, user_rating, feedback_comment, timestamp
            FROM query_feedback
            WHERE user_rating <= ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (threshold, limit))
        
        return [
            {
                'id': row['id'],
                'query': row['query'],
                'response': row['response'][:200] + '...',
                'rating': row['user_rating'],
                'comment': row['feedback_comment'],
                'timestamp': row['timestamp']
            }
            for row in cursor.fetchall()
        ]

def get_common_patterns(self, limit: int = 10) -> List[Dict]:
    """Analiza wspólnych wzorców w zapytaniach"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                LOWER(SUBSTR(query, 1, 50)) as pattern,
                COUNT(*) as frequency,
                AVG(user_rating) as avg_rating
            FROM query_feedback
            WHERE user_rating IS NOT NULL
            GROUP BY pattern
            HAVING frequency > 1
            ORDER BY frequency DESC
            LIMIT ?
        ''', (limit,))
        
        return [
            {
                'pattern': row['pattern'],
                'frequency': row['frequency'],
                'avg_rating': round(row['avg_rating'], 2) if row['avg_rating'] else None
            }
            for row in cursor.fetchall()
        ]

def export_feedback_report(self, filepath: str = "./feedback_report.json"):
    """Export pełnego raportu do JSON"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'analytics': self.get_analytics(),
        'low_rated_queries': self.get_low_rated_queries(),
        'common_patterns': self.get_common_patterns()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    self.logger.info(f"Report exported to {filepath}")
    return filepath
```

# ===========================

# 3. METADANE

# ===========================

class MetadataEnricher:
“”“Wzbogacanie dokumentów o metadane”””

```
@staticmethod
def enrich_document(doc, file_path: str):
    """Dodawanie metadanych do dokumentu"""
    file_stat = Path(file_path).stat()
    doc.metadata.update({
        'source': file_path,
        'file_size': file_stat.st_size,
        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
        'modified_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        'doc_hash': hashlib.md5(doc.page_content.encode()).hexdigest(),
        'version': '1.0'
    })
    return doc

@staticmethod
def add_chunk_metadata(doc, chunk_index: int, total_chunks: int):
    """Metadane dla chunków"""
    doc.metadata.update({
        'chunk_id': chunk_index,
        'total_chunks': total_chunks,
        'chunk_position': f"{chunk_index + 1}/{total_chunks}"
    })
    return doc
```

# ===========================

# 5. GŁÓWNA KLASA RAG Z A/B TESTING I AUTO-REINDEXING

# ===========================

class AdvancedRAGSystem:
“”“Zaawansowany system RAG z A/B testing i auto-reindexing”””

```
def __init__(self, config: RAGConfig = None, enable_ab_testing: bool = False):
    self.ab_testing_enabled = enable_ab_testing
    self.ab_manager = ABTestingManager() if enable_ab_testing else None
    self.reindex_manager = AutoReindexingManager()
    self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Przypisanie wariantu jeśli A/B testing włączony
    if self.ab_testing_enabled:
        variant_name = self.ab_manager.assign_variant(self.session_id)
        self.config = self.ab_manager.get_variant_config(variant_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Session {self.session_id} assigned to variant: {variant_name}")
    else:
        self.config = config or RAGConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    self.security = SecurityManager()
    self.metadata_enricher = MetadataEnricher()
    self.feedback = EnhancedFeedbackSystem()
    
    if not self.security.validate_api_key(self.config.openai_api_key):
        raise ValueError("Invalid OpenAI API key")
    
    os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
    
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)
    
    self.embeddings = None
    self.vectorstore = None
    self.qa_chain = None
    
    self._initialize_components()

def _initialize_components(self):
    """Inicjalizacja komponentów LangChain"""
    try:
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key,
            chunk_size=1000
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            request_timeout=60
        )
        
        self.logger.info(f"Components initialized (variant: {self.config.variant})")
        
    except Exception as e:
        self.logger.error(f"Initialization failed: {e}")
        raise

def load_and_process_documents(self, source_path: str, 
                               file_pattern: str = "**/*.txt") -> List:
    """Ładowanie i przetwarzanie dokumentów"""
    self.logger.info(f"Loading documents from: {source_path}")
    
    try:
        if os.path.isdir(source_path):
            loader = DirectoryLoader(
                source_path,
                glob=file_pattern,
                loader_cls=TextLoader,
                show_progress=True
            )
        else:
            loader = TextLoader(source_path)
        
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents")
        
        enriched_docs = []
        for doc in documents:
            enriched_doc = self.metadata_enricher.enrich_document(
                doc, doc.metadata.get('source', source_path)
            )
            enriched_docs.append(enriched_doc)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        texts = text_splitter.split_documents(enriched_docs)
        
        for idx, text in enumerate(texts):
            self.metadata_enricher.add_chunk_metadata(text, idx, len(texts))
        
        self.logger.info(f"Split into {len(texts)} chunks (size: {self.config.chunk_size})")
        return texts
        
    except Exception as e:
        self.logger.error(f"Document processing failed: {e}")
        self.feedback.log_error(
            self.session_id, 
            f"Document loading: {source_path}",
            "DocumentLoadError",
            str(e)
        )
        raise

def create_vectorstore(self, texts: List):
    """Tworzenie vector store"""
    try:
        persist_dir = f"{self.config.persist_directory}_{self.config.variant}"
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=persist_dir
        )
        self.logger.info(f"Vector store created (variant: {self.config.variant})")
        
    except Exception as e:
        self.logger.error(f"Vector store creation failed: {e}")
        self.feedback.log_error(
            self.session_id,
            "Vector store creation",
            "VectorStoreError",
            str(e)
        )
        raise

def load_vectorstore(self):
    """Ładowanie istniejącego vector store"""
    try:
        persist_dir = f"{self.config.persist_directory}_{self.config.variant}"
        
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.logger.info("Vector store loaded successfully")
        
    except Exception as e:
        self.logger.error(f"Vector store loading failed: {e}")
        raise

def setup_retriever(self):
    """Konfiguracja retrievera z parametrami wariantu"""
    if not self.vectorstore:
        raise ValueError("Vector store not initialized")
    
    search_kwargs = {
        "k": self.config.top_k,
        "fetch_k": self.config.top_k * 2
    }
    
    if self.config.search_type == "mmr":
        search_kwargs["lambda_mult"] = self.config.lambda_mult
    
    base_retriever = self.vectorstore.as_retriever(
        search_type=self.config.search_type,
        search_kwargs=search_kwargs
    )
    
    compressor = LLMChainExtractor.from_llm(self.llm)
    
    self.retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    self.logger.info(f"Retriever configured: {self.config.search_type}, k={self.config.top_k}")

def create_qa_chain(self):
    """Tworzenie łańcucha Q&A"""
    if not self.retriever:
        self.setup_retriever()
    
    self.qa_chain = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=self.retriever,
        return_source_documents=True,
        verbose=False
    )
    
    self.logger.info("QA chain created")

def query(self, question: str) -> Dict[str, Any]:
    """Główna metoda zapytania z pełnym trackingiem"""
    if not self.qa_chain:
        self.create_qa_chain()
    
    try:
        safe_question = self.security.sanitize_input(question)
        self.logger.info(f"Processing query: {safe_question[:100]}...")
        
        start_time = datetime.now()
        
        with get_openai_callback() as cb:
            result = self.qa_chain.invoke({"query": safe_question})
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Logowanie z wariantem A/B
            query_id = self.feedback.log_query(
                session_id=self.session_id,
                query=safe_question,
                response=result['result'],
                docs=result['source_documents'],
                response_time=response_time,
                tokens_used=cb.total_tokens,
                cost=cb.total_cost,
                model_used="gpt-4o-mini",
                variant_name=self.config.variant
            )
            
            return {
                'query_id': query_id,
                'answer': result['result'],
                'variant': self.config.variant,
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in result['source_documents']
                ],
                'metrics': {
                    'response_time': response_time,
                    'tokens_used': cb.total_tokens,
                    'total_cost': cb.total_cost
                }
            }
    
    except Exception as e:
        import traceback
        self.logger.error(f"Query failed: {e}")
        self.feedback.log_error(
            session_id=self.session_id,
            query=question,
            error_type=type(e).__name__,
            error_message=str(e),
            stack_trace=traceback.format_exc()
        )
        raise

def rate_response(self, query_id: int, rating: int, comment: str = None) -> bool:
    """Ocena odpowiedzi przez użytkownika"""
    return self.feedback.add_user_rating(query_id, rating, comment)

def get_system_stats(self, days: int = 30) -> Dict[str, Any]:
    """Statystyki systemu"""
    return self.feedback.get_analytics(days)

def get_ab_test_results(self) -> Dict[str, Any]:
    """Wyniki testów A/B"""
    if not self.ab_testing_enabled:
        return {"error": "A/B testing not enabled"}
    
    return self.ab_manager.get_variant_comparison()

def check_and_trigger_reindexing(self) -> Dict[str, Any]:
    """Sprawdzenie i uruchomienie auto-reindexing"""
    analysis = self.reindex_manager.analyze_feedback_for_reindexing()
    
    if analysis['needs_reindex']:
        self.logger.warning("Auto-reindexing triggered!")
        success = self.reindex_manager.auto_trigger_reindexing(self)
        analysis['reindex_executed'] = success
    else:
        analysis['reindex_executed'] = False
    
    return analysis

def get_improvement_suggestions(self) -> Dict[str, Any]:
    """Sugestie ulepszeń na podstawie feedbacku"""
    stats = self.get_system_stats()
    low_rated = self.feedback.get_low_rated_queries()
    patterns = self.feedback.get_common_patterns()
    
    suggestions = []
    
    if stats['avg_rating'] < 3.5:
        suggestions.append({
            'type': 'quality',
            'priority': 'high',
            'suggestion': 'Średnia ocena poniżej 3.5 - rozważ re-indexowanie lub zmianę wariantu'
        })
    
    if stats['failed_queries'] > stats['successful_queries'] * 0.1:
        suggestions.append({
            'type': 'reliability',
            'priority': 'high',
            'suggestion': f"Wysoki wskaźnik błędów ({stats['failed_queries']})"
        })
    
    if stats['avg_response_time'] > 5.0:
        suggestions.append({
            'type': 'performance',
            'priority': 'medium',
            'suggestion': 'Długi czas - zmniejsz chunk_size lub top_k'
        })
    
    # Sprawdź czy potrzebne re-indexowanie
    reindex_analysis = self.reindex_manager.analyze_feedback_for_reindexing()
    if reindex_analysis['needs_reindex']:
        suggestions.append({
            'type': 'reindexing',
            'priority': 'high',
            'suggestion': f"Re-indexing zalecany: {', '.join(reindex_analysis['reasons'])}"
        })
    
    return {
        'overall_health': 'good' if stats['avg_rating'] >= 4 else 'needs_improvement',
        'suggestions': suggestions,
        'low_rated_count': len(low_rated),
        'common_patterns_count': len(patterns),
        'reindex_needed': reindex_analysis['needs_reindex']
    }

def export_report(self, filepath: str = None):
    """Export raportu"""
    if filepath is None:
        filepath = f"./feedback_report_{self.session_id}.json"
    return self.feedback.export_feedback_report(filepath)
```

# ===========================

# 5. INTERAKTYWNY PRZYKŁAD UŻYCIA

# ===========================

def interactive_demo():
“”“Interaktywna demonstracja z oceną użytkownika”””

```
print("=" * 60)
print("ADVANCED RAG SYSTEM - INTERACTIVE DEMO")
print("=" * 60)

# Konfiguracja
config = RAGConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)

# Inicjalizacja
rag_system = AdvancedRAGSystem(config)

# Ładowanie danych
knowledge_base_path = "./knowledge_base"

if not os.path.exists(config.persist_directory):
    print("\n📚 Ładowanie dokumentów...")
    texts = rag_system.load_and_process_documents(knowledge_base_path)
    rag_system.create_vectorstore(texts)
else:
    print("\n📚 Ładowanie istniejącej bazy wiedzy...")
    rag_system.load_vectorstore()

rag_system.setup_retriever()
rag_system.create_qa_chain()

print("✅ System gotowy!\n")

# Pętla interaktywna
while True:
    print("\n" + "=" * 60)
    question = input("❓ Twoje pytanie (lub 'stats' / 'report' / 'quit'): ").strip()
    
    if question.lower() == 'quit':
        break
    
    if question.lower() == 'stats':
        stats = rag_system.get_system_stats()
        print("\n📊 STATYSTYKI SYSTEMU:")
        print("-" * 60)
        for key, value in stats.items():
            if key != 'rating_distribution' and key != 'top_errors':
                print(f"{key}: {value}")
        
        if stats['rating_distribution']:
            print("\n⭐ Rozkład ocen:")
            for rating, count in sorted(stats['rating_distribution'].items()):
                print(f"  {rating} gwiazdek: {count} ocen")
        
        if stats['top_errors']:
            print("\n❌ Najczęstsze błędy:")
            for error in stats['top_errors']:
                print(f"  {error['error_type']}: {error['count']} razy")
        
        # Sugestie ulepszeń
        suggestions = rag_system.get_improvement_suggestions()
        print(f"\n💡 Stan systemu: {suggestions['overall_health']}")
        if suggestions['suggestions']:
            print("Sugestie ulepszeń:")
            for sug in suggestions['suggestions']:
                print(f"  [{sug['priority']}] {sug['suggestion']}")
        
        continue
    
    if question.lower() == 'report':
        filepath = rag_system.export_report()
        print(f"\n📄 Raport wyeksportowany do: {filepath}")
        continue
    
    if not question:
        continue
    
    # Zapytanie
    try:
        print("\n🤔 Przetwarzam...")
        result = rag_system.query(question)
        
        print("\n" + "=" * 60)
        print("💬 ODPOWIEDŹ:")
        print("=" * 60)
        print(result['answer'])
        
        print("\n" + "=" * 60)
        print("📚 ŹRÓDŁA:")
        print("=" * 60)
        for idx, doc in enumerate(result['source_documents'], 1):
            print(f"\n{idx}. {doc['metadata'].get('source', 'Unknown')}")
            print(f"   Chunk: {doc['metadata'].get('chunk_position', 'N/A')}")
            print(f"   {doc['content'][:150]}...")
        
        print("\n" + "=" * 60)
        print("⚡ METRYKI:")
        print("=" * 60)
        print(f"Czas odpowiedzi: {result['metrics']['response_time']:.2f}s")
        print(f"Tokeny: {result['metrics']['tokens_used']}")
        print(f"Koszt: ${result['metrics']['total_cost']:.4f}")
        
        # Prośba o ocenę
        print("\n" + "=" * 60)
        print("⭐ OCEŃ ODPOWIEDŹ")
        print("=" * 60)
        
        while True:
            rating_input = input("Oceń odpowiedź (1-5 gwiazdek, lub enter aby pominąć): ").strip()
            
            if not rating_input:
                print("Pominięto ocenę.")
                break
            
            try:
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    # Opcjonalny komentarz
                    comment = input("Dodatkowy komentarz (opcjonalnie): ").strip()
                    comment = comment if comment else None
                    
                    # Zapisanie oceny
                    success = rag_system.rate_response(
                        result['query_id'], 
                        rating, 
                        comment
                    )
                    
                    if success:
                        print(f"✅ Dziękujemy! Zapisano ocenę: {rating} ⭐")
                        
                        # Reakcja na niską ocenę
                        if rating <= 2:
                            print("😔 Przykro nam, że odpowiedź nie spełniła Twoich oczekiwań.")
                            print("   Twój feedback pomoże nam ulepszyć system!")
                    else:
                        print("❌ Błąd przy zapisywaniu oceny.")
                    break
                else:
                    print("❌ Ocena musi być w zakresie 1-5")
            except ValueError:
                print("❌ Proszę wpisać liczbę od 1 do 5")
    
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        print("Błąd został zapisany do analizy.")

# Podsumowanie sesji
print("\n" + "=" * 60)
print("📊 PODSUMOWANIE SESJI")
print("=" * 60)

stats = rag_system.get_system_stats(days=1)
print(f"Zapytań w tej sesji: {stats['total_queries']}")
print(f"Średnia ocena: {stats['avg_rating']}/5.0")
print(f"Całkowity koszt: ${stats['total_cost']:.4f}")

print("\n👋 Dziękujemy za korzystanie z systemu!")
```

def automated_testing_demo():
“”“Demo z automatycznymi testami i analizą”””

```
print("=" * 60)
print("AUTOMATED TESTING & ANALYSIS DEMO")
print("=" * 60)

config = RAGConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)

rag_system = AdvancedRAGSystem(config)

# Przykładowe zapytania testowe
test_queries = [
    {
        'query': 'What is machine learning?',
        'expected_rating': 5,
        'comment': 'Clear and comprehensive answer'
    },
    {
        'query': 'Explain quantum computing',
        'expected_rating': 4,
        'comment': 'Good explanation but could be simpler'
    },
    {
        'query': 'What is the weather today?',
        'expected_rating': 1,
        'comment': 'Question not relevant to knowledge base'
    }
]

print("\n🧪 Uruchamianie testów...\n")

for idx, test in enumerate(test_queries, 1):
    print(f"Test {idx}/{len(test_queries)}: {test['query']}")
    
    try:
        result = rag_system.query(test['query'])
        
        # Automatyczna ocena
        rag_system.rate_response(
            result['query_id'],
            test['expected_rating'],
            test['comment']
        )
        
        print(f"  ✅ Odpowiedź: {result['answer'][:100]}...")
        print(f"  ⭐ Ocena: {test['expected_rating']}/5")
        print(f"  ⚡ Czas: {result['metrics']['response_time']:.2f}s\n")
        
    except Exception as e:
        print(f"  ❌ Błąd: {e}\n")

# Analiza wyników
print("=" * 60)
print("📊 ANALIZA WYNIKÓW TESTÓW")
print("=" * 60)

stats = rag_system.get_system_stats()

print(f"\nPodstawowe statystyki:")
print(f"  Wykonanych zapytań: {stats['total_queries']}")
print(f"  Udanych: {stats['successful_queries']}")
print(f"  Nieudanych: {stats['failed_queries']}")
print(f"  Wskaźnik sukcesu: {stats['success_rate']:.1f}%")

print(f"\nJakość odpowiedzi:")
print(f"  Średnia ocena: {stats['avg_rating']:.2f}/5.0")
print(f"  Pozytywne oceny (4-5★): {stats['positive_ratings']}")
print(f"  Negatywne oceny (1-2★): {stats['negative_ratings']}")

print(f"\nWydajność:")
print(f"  Średni czas odpowiedzi: {stats['avg_response_time']:.2f}s")
print(f"  Całkowite tokeny: {stats['total_tokens']}")
print(f"  Całkowity koszt: ${stats['total_cost']:.4f}")

# Zapytania do poprawy
print("\n" + "=" * 60)
print("🔍 ZAPYTANIA WYMAGAJĄCE UWAGI")
print("=" * 60)

low_rated = rag_system.feedback.get_low_rated_queries(threshold=3, limit=5)

if low_rated:
    for query in low_rated:
        print(f"\n❌ Ocena: {query['rating']}/5")
        print(f"   Pytanie: {query['query']}")
        print(f"   Odpowiedź: {query['response']}")
        if query['comment']:
            print(f"   Komentarz: {query['comment']}")
        print(f"   Data: {query['timestamp']}")
else:
    print("\n✅ Brak zapytań z niską oceną!")

# Sugestie ulepszeń
print("\n" + "=" * 60)
print("💡 SUGESTIE ULEPSZEŃ")
print("=" * 60)

suggestions = rag_system.get_improvement_suggestions()

print(f"\nOgólny stan: {suggestions['overall_health'].upper()}")

if suggestions['suggestions']:
    print("\nZalecane działania:")
    for sug in suggestions['suggestions']:
        priority_emoji = "🔴" if sug['priority'] == 'high' else "🟡"
        print(f"{priority_emoji} [{sug['type'].upper()}] {sug['suggestion']}")
else:
    print("\n✅ System działa optymalnie!")

# Export raportu
print("\n" + "=" * 60)
report_path = rag_system.export_report()
print(f"📄 Szczegółowy raport zapisany: {report_path}")
print("=" * 60)
```

# ===========================

# 6. DASHBOARD ANALYTICS (BONUS)

# ===========================

def generate_analytics_dashboard():
“”“Generowanie dashboard’u analitycznego”””

```
feedback = EnhancedFeedbackSystem()

print("=" * 60)
print("📊 ANALYTICS DASHBOARD")
print("=" * 60)

# Statystyki dla różnych okresów
periods = [7, 30, 90]

for days in periods:
    print(f"\n{'─' * 60}")
    print(f"📅 OSTATNIE {days} DNI")
    print('─' * 60)
    
    stats = feedback.get_analytics(days)
    
    print(f"\n✅ Sukces: {stats['success_rate']:.1f}%")
    print(f"   └─ Udane: {stats['successful_queries']}")
    print(f"   └─ Błędy: {stats['failed_queries']}")
    
    print(f"\n⭐ Jakość: {stats['avg_rating']:.2f}/5.0")
    print(f"   └─ Pozytywne (4-5★): {stats['positive_ratings']}")
    print(f"   └─ Negatywne (1-2★): {stats['negative_ratings']}")
    
    if stats['rating_distribution']:
        print(f"\n   Rozkład ocen:")
        total_ratings = sum(stats['rating_distribution'].values())
        for rating in range(5, 0, -1):
            count = stats['rating_distribution'].get(rating, 0)
            percentage = (count / total_ratings * 100) if total_ratings > 0 else 0
            bar = '█' * int(percentage / 5)
            print(f"   {rating}★ {bar} {percentage:.1f}% ({count})")
    
    print(f"\n⚡ Wydajność:")
    print(f"   └─ Średni czas: {stats['avg_response_time']:.2f}s")
    print(f"   └─ Tokeny: {stats['total_tokens']:,}")
    
    print(f"\n💰 Koszt: ${stats['total_cost']:.4f}")

# Najczęstsze wzorce
print(f"\n{'=' * 60}")
print("🔍 NAJCZĘSTSZE WZORCE ZAPYTAŃ")
print('=' * 60)

patterns = feedback.get_common_patterns(limit=10)

if patterns:
    for idx, pattern in enumerate(patterns, 1):
        rating_indicator = "✅" if pattern['avg_rating'] and pattern['avg_rating'] >= 4 else "⚠️"
        print(f"\n{idx}. {rating_indicator} Wystąpienia: {pattern['frequency']}")
        print(f"   Wzorzec: {pattern['pattern']}")
        if pattern['avg_rating']:
            print(f"   Średnia ocena: {pattern['avg_rating']:.1f}/5.0")
else:
    print("\nBrak wystarczających danych do analizy wzorców.")

print("\n" + "=" * 60)
```

# ===========================

# 7. MAIN

# ===========================

if **name** == “**main**”:
import sys

```
print("\n🚀 ADVANCED RAG SYSTEM WITH USER FEEDBACK\n")
print("Wybierz tryb:")
print("1. Interaktywny (rozmowa z systemem)")
print("2. Automatyczne testy")
print("3. Dashboard analityczny")
print("4. Wyjście\n")

choice = input("Wybór (1-4): ").strip()

if choice == "1":
    interactive_demo()
elif choice == "2":
    automated_testing_demo()
elif choice == "3":
    generate_analytics_dashboard()
else:
    print("👋 Do zobaczenia!")
```
