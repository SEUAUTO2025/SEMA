"""
@filename: AI_dbmanager.py
@description: AI Database Operations Class
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# absolute_project_root = r"D:\Pythonworks\SpatialTemporalAttentionGCN-master\SpatialTemporalAttentionGCN-master"
#
# project_root = os.path.abspath(absolute_project_root)
#
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from sqlalchemy import Column, Integer, String, Text, LargeBinary, Float, ForeignKey, MetaData, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from openai import OpenAI
try:
    from deep_translator import DeeplTranslator
except ImportError:
    DeeplTranslator = None
import pickle
from RAG.Knowledge_Database.AIdbconfig import engine, session, db_pure_path
import time
import numpy as np
import re

rag_split_and_merge = None
_rag_get_embedding_languagebind_video = None
_rag_get_embedding_languagebind_text = None


def _load_ragfunc_helpers():
    global rag_split_and_merge
    global _rag_get_embedding_languagebind_video
    global _rag_get_embedding_languagebind_text

    if (
        rag_split_and_merge is not None
        and _rag_get_embedding_languagebind_video is not None
        and _rag_get_embedding_languagebind_text is not None
    ):
        return

    from RAG.Knowledge_Database.RAGFunc import (
        split_and_merge as imported_split_and_merge,
        get_embedding_languagebind_video as imported_get_embedding_languagebind_video,
        get_embedding_languagebind_text as imported_get_embedding_languagebind_text,
    )

    rag_split_and_merge = imported_split_and_merge
    _rag_get_embedding_languagebind_video = imported_get_embedding_languagebind_video
    _rag_get_embedding_languagebind_text = imported_get_embedding_languagebind_text


def get_embedding(texts,model_name="text-embedding-v4"):
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.embeddings.create(
        model=model_name,
        input=texts,
        dimensions=1024,
        encoding_format="float"
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def get_embedding_languagebind_video(video_path):
    _load_ragfunc_helpers()
    return _rag_get_embedding_languagebind_video(video_path)


def get_embedding_languagebind_text(texts):
    _load_ragfunc_helpers()
    return _rag_get_embedding_languagebind_text(texts)

Base = declarative_base()

class Document(Base):
    """Document table"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200))
    content = Column(Text)
    title_embedding = Column(LargeBinary)

    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    """Document chunk table"""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text)

    embedding = Column(LargeBinary)  # Store embedding vector (serialized into BLOB)

    document = relationship("Document", back_populates="chunks")
    
class Embedding(Base):
    """Embedding table for storing text vector representations"""
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    vector = Column(LargeBinary)  # Store serialized vector
    model_name = Column(String(100))  # Record the embedding model used
    created_at = Column(Float)  # Record creation time

    chunk = relationship("Chunk", back_populates="embeddings")

Chunk.embeddings = relationship("Embedding", back_populates="chunk")

SECTION_TITLE_RE = re.compile(r"(?m)^//\s*(.+?)\s*$")


def ensure_database_schema(db_engine):
    """Create missing tables and add title embedding column for legacy DBs."""
    Base.metadata.create_all(db_engine)
    inspector = inspect(db_engine)
    table_names = set(inspector.get_table_names())
    if "documents" not in table_names:
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")}
    if "title_embedding" not in document_columns:
        with db_engine.begin() as conn:
            conn.execute(text("ALTER TABLE documents ADD COLUMN title_embedding BLOB"))


def split_and_merge(text, chunk_size=500):
    """Use the repo splitter when available; otherwise fall back to a lightweight local splitter."""
    if rag_split_and_merge is None:
        try:
            _load_ragfunc_helpers()
        except Exception:
            pass

    if rag_split_and_merge is not None:
        return rag_split_and_merge(text, chunk_size=chunk_size)

    normalized_text = str(text or "").replace("\r\n", "\n").strip()
    if not normalized_text:
        return []

    paragraphs = [p.strip() for p in normalized_text.split("\n\n") if p.strip()]
    chunks = []
    sentence_re = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")

    for paragraph in paragraphs:
        sentences = [s.strip() for s in sentence_re.findall(paragraph) if s.strip()]
        if not sentences:
            sentences = [paragraph]

        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= chunk_size:
                current += sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

    return chunks


def parse_structured_knowledge_sections(content):
    """Parse `//title` knowledge blocks from zhishi-style text."""
    text = str(content or "").replace("\r\n", "\n")
    matches = list(SECTION_TITLE_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if title and body:
            sections.append((title, body))
    return sections


def build_section_chunks(title, body, chunk_size):
    """Split one knowledge section into plain text chunks without duplicating the title."""
    normalized_title = str(title or "").strip()
    normalized_body = str(body or "").strip()
    if not normalized_title or not normalized_body:
        return []

    section_chunks = split_and_merge(normalized_body, chunk_size=chunk_size)
    if not section_chunks:
        section_chunks = [normalized_body]

    plain_chunks = []
    for chunk_text in section_chunks:
        normalized_chunk = str(chunk_text or "").strip()
        if normalized_chunk:
            plain_chunks.append(normalized_chunk)
    return plain_chunks


def build_document_chunks(title, content, chunk_size):
    """Chunk structured KB text per `//title` block before sentence chunking."""
    sections = parse_structured_knowledge_sections(content)
    if not sections:
        return build_section_chunks(title=title, body=content, chunk_size=chunk_size)

    chunks = []
    for section_title, section_body in sections:
        chunks.extend(build_section_chunks(title=section_title, body=section_body, chunk_size=chunk_size))
    return chunks


def build_pure_knowledge_db(zhishi_path, db_path, chunk_size=260, rebuild=True):
    """Build the pure knowledge DB from zhishi.txt using one document per knowledge block."""
    zhishi_path = os.path.abspath(zhishi_path)
    db_path = os.path.abspath(db_path)

    if not os.path.exists(zhishi_path):
        raise FileNotFoundError(f"Knowledge source file not found: {zhishi_path}")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    pure_engine = create_engine(f"sqlite:///{db_path}", echo=False)
    pure_session_factory = sessionmaker(bind=pure_engine)
    ensure_database_schema(pure_engine)
    pure_session = pure_session_factory()

    if rebuild and os.path.exists(db_path):
        try:
            pure_session.close()
            pure_engine.dispose()
            os.remove(db_path)
            pure_engine = create_engine(f"sqlite:///{db_path}", echo=False)
            pure_session_factory = sessionmaker(bind=pure_engine)
            ensure_database_schema(pure_engine)
            pure_session = pure_session_factory()
        except PermissionError:
            pure_engine = create_engine(f"sqlite:///{db_path}", echo=False)
            pure_session_factory = sessionmaker(bind=pure_engine)
            pure_session = pure_session_factory()
            ensure_database_schema(pure_engine)
            pure_session.query(Embedding).delete()
            pure_session.query(Chunk).delete()
            pure_session.query(Document).delete()
            pure_session.commit()

    with open(zhishi_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sections = parse_structured_knowledge_sections(raw_text)
    if not sections:
        raise ValueError(f"No structured `//title` sections found in {zhishi_path}")

    document_count = 0
    chunk_count = 0

    try:
        for section_title, section_body in sections:
            document = Document(title=section_title, content=section_body)
            pure_session.add(document)
            pure_session.flush()

            section_chunks = build_section_chunks(
                title=section_title,
                body=section_body,
                chunk_size=chunk_size,
            )
            for chunk_text in section_chunks:
                pure_session.add(
                    Chunk(
                        document_id=document.id,
                        text=chunk_text,
                    )
                )

            document_count += 1
            chunk_count += len(section_chunks)

        pure_session.commit()
        return {
            "db_path": db_path,
            "zhishi_path": zhishi_path,
            "sections": len(sections),
            "documents": document_count,
            "chunks": chunk_count,
            "embeddings": 0,
            "chunk_size": chunk_size,
        }
    except Exception:
        pure_session.rollback()
        raise
    finally:
        pure_session.close()
        pure_engine.dispose()

def batch_translate_files(input_folder, api_key, target_lang='en'):
    """Batch translate text files using DeepL API"""
    if DeeplTranslator is None:
        raise ImportError("deep_translator is required for batch_translate_files but is not installed.")

    translator = DeeplTranslator(api_key=api_key, source="zh", target=target_lang, use_free_api=True)
    
    output_folder = os.path.join(input_folder, "translated_results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            
            print(f"Processing: {filename}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().replace('\n', ' ').strip()
                
                if not content:
                    continue

                translated_text = translator.translate(content)
                
                output_path = os.path.join(output_folder, f"translated_{filename}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
                
                print(f"Successfully exported: translated_{filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

class KnowledgeDB:
    def __init__(self, session):
        self.engine = session.bind if getattr(session, "bind", None) is not None else engine
        self.session = session
        self._ensure_database_schema()

    def _ensure_database_schema(self):
        """Create missing tables and add title embedding column for legacy DBs."""
        ensure_database_schema(self.engine)

    @staticmethod
    def _vector_to_bytes(vector):
        return pickle.dumps(np.asarray(vector, dtype=np.float32))

    @staticmethod
    def _vector_from_bytes(blob):
        return np.asarray(pickle.loads(blob), dtype=np.float32).flatten()

    @staticmethod
    def _cosine_similarity(vec, query_vec):
        vec = np.asarray(vec, dtype=np.float32).flatten()
        query_vec = np.asarray(query_vec, dtype=np.float32).flatten()
        denom = np.linalg.norm(vec) * np.linalg.norm(query_vec)
        if denom <= 0:
            return -1.0
        return float(np.dot(vec, query_vec) / denom)

    @staticmethod
    def _normalize_query_vectors(query_vec):
        if isinstance(query_vec, np.ndarray):
            raw_vectors = [query_vec] if query_vec.ndim <= 1 else list(query_vec)
        elif isinstance(query_vec, (list, tuple)):
            if not query_vec:
                raw_vectors = []
            elif isinstance(query_vec[0], (list, tuple, np.ndarray)):
                raw_vectors = list(query_vec)
            else:
                raw_vectors = [query_vec]
        else:
            raw_vectors = [query_vec]

        normalized_vectors = []
        for item in raw_vectors:
            vec = np.asarray(item, dtype=np.float32).flatten()
            if vec.size == 0:
                continue
            if np.linalg.norm(vec) <= 0:
                continue
            normalized_vectors.append(vec)
        return normalized_vectors

    def _get_query_vector(self, query, embed_fn):
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("query must not be empty")
        return np.asarray(embed_fn([query_text])[0], dtype=np.float32).flatten()

    def _get_chunk_embeddings_map(self, model_name="default", document_ids=None):
        query = self.session.query(Chunk, Embedding).join(
            Embedding, Embedding.chunk_id == Chunk.id
        ).filter(Embedding.model_name == model_name)

        if document_ids is not None:
            doc_ids = [int(doc_id) for doc_id in document_ids]
            if not doc_ids:
                return {}
            query = query.filter(Chunk.document_id.in_(doc_ids))

        embeddings_map = {}
        for chunk, embedding in query.all():
            existing = embeddings_map.get(chunk.id)
            if existing is None or embedding.created_at >= existing[1].created_at:
                embeddings_map[chunk.id] = (chunk, embedding)
        return embeddings_map
        
    def add_document(self, title, content, embed_fn=None, model_name="default", chunk_size=260):
        """Add document and split into chunks"""
        doc = Document(title=title, content=content)
        self.session.add(doc)
        self.session.commit()

        chunks = build_document_chunks(title=title, content=content, chunk_size=chunk_size)

        for text in chunks:
            chunk = Chunk(
                document_id=doc.id,
                text=text
            )
            self.session.add(chunk)

        self.session.commit()
        return doc

    def clear_chunks(self):
        """Clear all chunks and related embeddings"""
        try:
            self.session.query(Embedding).delete()
            self.session.query(Chunk).delete()
            self.session.commit()
            print("All chunks and embeddings cleared")
        except Exception as e:
            self.session.rollback()
            print(f"Error clearing chunks: {e}")

    def scan_and_add_chunks(self, chunk_size=260):
        """Scan all documents and regenerate chunks"""
        try:
            documents = self.session.query(Document).all()
            print(f"Found {len(documents)} documents, generating chunks...")
            
            count = 0
            for doc in documents:
                if not doc.content:
                    continue
                    
                chunks = build_document_chunks(title=doc.title, content=doc.content, chunk_size=chunk_size)
                for text in chunks:
                    chunk = Chunk(
                        document_id=doc.id,
                        text=text
                    )
                    self.session.add(chunk)
                    count += 1
            
            self.session.commit()
            print(f"Generated {count} chunks for {len(documents)} documents")
        except Exception as e:
            self.session.rollback()
            print(f"Error generating chunks: {e}")
    
    def delete_document(self, document_id: int):
        """Delete document and its related chunks and embeddings"""
        doc = self.session.query(Document).filter(Document.id == document_id).first()
        if not doc:
            print(f"Document {document_id} does not exist")
            return False
            
        chunks = self.session.query(Chunk).filter(Chunk.document_id == document_id).all()
        for chunk in chunks:
            self.session.query(Embedding).filter(Embedding.chunk_id == chunk.id).delete()
            
        self.session.query(Chunk).filter(Chunk.document_id == document_id).delete()
        self.session.delete(doc)
        self.session.commit()
        return True
    
    def delete_embedding(self):
        """Delete all embeddings"""
        embeddings = self.session.query(Embedding).all()
        for embedding in embeddings:
            self.session.delete(embedding)
        self.session.query(Chunk).update({Chunk.embedding: None}, synchronize_session=False)
        self.session.commit()
        return True

    def sync_chunk_embedding_column_from_embeddings(self, model_name="default"):
        """
        Temporary helper: mirror the latest vector in `embeddings` back into `chunks.embedding`.
        Because `chunks.embedding` is a single column, it only stores one snapshot per chunk.
        """
        print(f"Syncing chunk.embedding from embeddings (model={model_name})...")
        embeddings_map = self._get_chunk_embeddings_map(model_name=model_name)
        if not embeddings_map:
            print("No matching embeddings found to sync.")
            return 0

        updated_count = 0
        try:
            for chunk, embedding in embeddings_map.values():
                chunk.embedding = embedding.vector
                updated_count += 1
            self.session.commit()
            print(f"Synchronized {updated_count} chunk embeddings.")
            return updated_count
        except Exception:
            self.session.rollback()
            raise

    def update_embeddings(self, embed_fn, model_name="default", batch_size=10):
        """Update title embeddings for documents and text embeddings for chunks."""
        if embed_fn is None:
            raise ValueError("embed_fn must not be None")
        batch_size = max(1, min(int(batch_size), 10))

        print("Starting to update title and chunk embeddings...")
        documents = self.session.query(Document).all()
        chunks = self.session.query(Chunk).all()
        successful_title_batches = 0
        successful_chunk_batches = 0

        if not documents and not chunks:
            print("No documents or chunks found to update.")
            return

        self.session.query(Embedding).filter(Embedding.model_name == model_name).delete()
        self.session.query(Document).update({Document.title_embedding: None}, synchronize_session=False)
        self.session.query(Chunk).update({Chunk.embedding: None}, synchronize_session=False)
        self.session.commit()

        if documents:
            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                titles_to_embed = [str(doc.title or "").strip() for doc in batch_documents]
                print(f"Processing title batch {i // batch_size + 1}: Sending {len(titles_to_embed)} titles to the embedding model...")
                try:
                    title_vectors = embed_fn(titles_to_embed)
                    if len(title_vectors) != len(batch_documents):
                        raise ValueError(
                            f"Mismatch in count for title batch: Received {len(title_vectors)} embeddings for {len(batch_documents)} documents."
                        )
                    for document, vector in zip(batch_documents, title_vectors):
                        document.title_embedding = self._vector_to_bytes(vector)
                    self.session.commit()
                    successful_title_batches += 1
                except Exception as e:
                    print(f"An error occurred during title batch {i // batch_size + 1}: {e}")
                    self.session.rollback()
                    continue

        if not chunks:
            print("No chunks found to update after title embedding generation.")
            return

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts_to_embed = [chunk.text for chunk in batch_chunks]

            print(f"Processing chunk batch {i // batch_size + 1}: Sending {len(texts_to_embed)} text chunks to the embedding model...")
            try:
                embedding_vectors = embed_fn(texts_to_embed)

                if len(embedding_vectors) != len(batch_chunks):
                    raise ValueError(
                        f"Mismatch in count for batch: Received {len(embedding_vectors)} embeddings for {len(batch_chunks)} chunks."
                    )

                for chunk, vector in zip(batch_chunks, embedding_vectors):
                    vector_blob = self._vector_to_bytes(vector)
                    chunk.embedding = vector_blob
                    embedding = Embedding(
                        chunk=chunk,
                        vector=vector_blob,
                        model_name=model_name,
                        created_at=time.time()
                    )
                    self.session.add(embedding)
                self.session.commit()
                successful_chunk_batches += 1

            except Exception as e:
                print(f"An error occurred during chunk batch {i // batch_size + 1}: {e}")
                self.session.rollback()
                continue

        if documents and successful_title_batches == 0:
            raise RuntimeError("Failed to generate any document title embeddings.")
        if chunks and successful_chunk_batches == 0:
            raise RuntimeError("Failed to generate any chunk embeddings.")

        print("All embeddings updated successfully.")

    def clear_database(self):
        """Clear all data in database tables without deleting table structure"""
        meta = MetaData()
        meta.reflect(bind=self.engine)
        with self.session.begin():
            for table in reversed(meta.sorted_tables):
                self.session.execute(table.delete())
        self.session.commit()
        print("Database cleared")
        
    def search(self, query, embed_fn, model_name="default", top_k=3):
        """Search for similar content"""
        query_vec = self._get_query_vector(query, embed_fn)
        results = []

        for chunk, embedding in self._get_chunk_embeddings_map(model_name=model_name).values():
            score = self._cosine_similarity(self._vector_from_bytes(embedding.vector), query_vec)
            if score > 0.5:
                results.append((chunk, score))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_document_titles(self, query, embed_fn, title_top_k=3):
        """Search similar document titles using cached title embeddings."""
        query_vec = self._get_query_vector(query, embed_fn)
        results = []
        documents = self.session.query(Document).filter(Document.title_embedding.isnot(None)).all()

        for document in documents:
            score = self._cosine_similarity(self._vector_from_bytes(document.title_embedding), query_vec)
            results.append((document, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:title_top_k]

    def search_chunks_within_documents(self, query, document_ids, embed_fn, model_name="default", chunk_top_k=5):
        """Search chunks only within the given document ids."""
        query_vec = self._get_query_vector(query, embed_fn)
        results = []

        for chunk, embedding in self._get_chunk_embeddings_map(model_name=model_name, document_ids=document_ids).values():
            score = self._cosine_similarity(self._vector_from_bytes(embedding.vector), query_vec)
            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:chunk_top_k]

    def search_knowledge_two_stage(self, query, embed_fn, model_name="default", title_top_k=5, chunk_top_k=8):
        """Retrieve top titles first, then search chunks inside those title-selected documents."""
        title_results = self.search_document_titles(query, embed_fn=embed_fn, title_top_k=title_top_k)
        document_ids = [document.id for document, _ in title_results]
        
        if not document_ids:
            return self.search(query, embed_fn=embed_fn, model_name=model_name, top_k=chunk_top_k)

        chunk_results = self.search_chunks_within_documents(
            query=query,
            document_ids=document_ids,
            embed_fn=embed_fn,
            model_name=model_name,
            chunk_top_k=chunk_top_k,
        )
        if chunk_results:
            return chunk_results
        return self.search(query, embed_fn=embed_fn, model_name=model_name, top_k=chunk_top_k)

    def from_video_search(self, query_vec, model_name="default", top_k=3):
        """Search similar chunks from one or more query vectors."""
        query_vec_list = self._normalize_query_vectors(query_vec)
        if not query_vec_list:
            return []

        chunk_scores = {}
        for chunk, embedding in self._get_chunk_embeddings_map(model_name=model_name).values():
            vec = self._vector_from_bytes(embedding.vector)
            best_score = None

            for single_query_vec in query_vec_list:
                score = self._cosine_similarity(vec, single_query_vec)
                if best_score is None or score > best_score:
                    best_score = score

            if best_score is not None and best_score > 0.1:
                chunk_scores[chunk.text] = max(chunk_scores.get(chunk.text, -1.0), best_score)

        results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in results[:top_k]]
    
    def embedding_model_choose(self, model):
        """Choose embedding model"""
        if model == "ali-text-embedding-v3":
            return get_embedding
        elif model == 'languagebind_video':
            return get_embedding_languagebind_video
        elif model == 'languagebind_text':
            return get_embedding_languagebind_text
        
    
    def chat(self, user_question: str, top_k=3, embedding_model="ali-text-embedding-v3", chat_model="deepseek"):
        """
        Handle chat logic: search context, call LLM, and print response
        """
        print(f"Received question: {user_question}")
        print("Retrieving relevant information...")

        try:
            search_results = self.search(
                user_question,
                embed_fn=self.embedding_model_choose(embedding_model),
                model_name=embedding_model,
                top_k=top_k
            )
        except Exception as e:
            print(f"Error during information retrieval: {e}")
            return

        if not search_results:
            print("No relevant information found.")
            return

        context_chunks = [chunk.text for chunk, score in search_results]
        context = "\n\n---\n\n".join(context_chunks)

        print("Context found, generating response...")
        
        try:
            response = self.chat_model_choose(chat_model)(prompt=user_question, context=context)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return

        print("\nModel response:")
        print(response)
        
# if __name__ == "__main__":
#     rag_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     zhishi_path = os.path.join(rag_root, "zhishi.txt")
#     stats = build_pure_knowledge_db(
#         zhishi_path=zhishi_path,
#         db_path=db_pure_path,
#         chunk_size=200,
#         rebuild=True,
#     )

#     print("Pure knowledge DB rebuild complete.")
#     print(f"Source: {stats['zhishi_path']}")
#     print(f"Target DB: {stats['db_path']}")
#     print(f"Sections: {stats['sections']}")
#     print(f"Documents: {stats['documents']}")
#     print(f"Chunks: {stats['chunks']}")
#     print(f"Embeddings: {stats['embeddings']}")
# if __name__ == "__main__":
    # from RAG.Knowledge_Database.AIdbconfig import session_pure

    # db = KnowledgeDB(session_pure)
    # # Temporary one-off backfill for existing rows, if needed:
    # # db.sync_chunk_embedding_column_from_embeddings(model_name="ali-text-embedding-v4")
    # db.update_embeddings(
    #     embed_fn=get_embedding,
    #     model_name="ali-text-embedding-v4",
    #     batch_size=10,
    # )
