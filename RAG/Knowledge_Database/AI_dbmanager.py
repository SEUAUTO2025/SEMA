"""
@filename: AI_dbmanager.py
@description: AI Database Operations Class
"""

import sys, os
absolute_project_root = r"D:\Pythonworks\SpatialTemporalAttentionGCN-master\SpatialTemporalAttentionGCN-master"

project_root = os.path.abspath(absolute_project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import Column, Integer, String, Text, LargeBinary, Float, ForeignKey, MetaData, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import numpy as np
import pickle
from AIdbconfig import engine, session
import time
from RAGFunc import *

Base = declarative_base()

class Document(Base):
    """Document table"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200))
    content = Column(Text)

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

import os
from deep_translator import DeeplTranslator

def batch_translate_files(input_folder, api_key, target_lang='en'):
    """Batch translate text files using DeepL API"""
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
        self.engine = engine
        self.session = session
        
    def add_document(self, title, content, embed_fn=None, model_name="default", chunk_size=70):
        """Add document and split into chunks"""
        doc = Document(title=title, content=content)
        self.session.add(doc)
        self.session.commit()

        chunks = split_and_merge(content, chunk_size=chunk_size)

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

    def scan_and_add_chunks(self, chunk_size=70):
        """Scan all documents and regenerate chunks"""
        try:
            documents = self.session.query(Document).all()
            print(f"Found {len(documents)} documents, generating chunks...")
            
            count = 0
            for doc in documents:
                if not doc.content:
                    continue
                    
                chunks = split_and_merge(doc.content, chunk_size=chunk_size)
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
        self.session.commit()
        return True

    def update_embeddings(self, embed_fn, model_name="default", batch_size=10):
        """Update embeddings for all chunks"""
        print("Starting to update embeddings for all chunks...")
        chunks = self.session.query(Chunk).all()

        if not chunks:
            print("No chunks found to update.")
            return

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts_to_embed = [chunk.text for chunk in batch_chunks]

            print(f"Processing batch {i // batch_size + 1}: Sending {len(texts_to_embed)} text chunks to the embedding model...")

            try:
                embedding_vectors = embed_fn(texts_to_embed)

                if len(embedding_vectors) != len(batch_chunks):
                    raise ValueError(
                        f"Mismatch in count for batch: Received {len(embedding_vectors)} embeddings for {len(batch_chunks)} chunks."
                    )

                for chunk, vector in zip(batch_chunks, embedding_vectors):
                    embedding = Embedding(
                        chunk=chunk,
                        vector=pickle.dumps(vector),
                        model_name=model_name,
                        created_at=time.time()
                    )
                    self.session.add(embedding)

            except Exception as e:
                print(f"An error occurred during batch {i // batch_size + 1}: {e}")
                self.session.rollback()
                continue

        self.session.commit()
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
        query_vec = embed_fn([query])[0]
        results = []
        
        chunks = self.session.query(Chunk).join(Embedding).filter(
            Embedding.model_name == model_name
        ).all()
        
        for chunk in chunks:
            embedding = self.session.query(Embedding).filter(
                Embedding.chunk_id == chunk.id,
                Embedding.model_name == model_name
            ).order_by(Embedding.created_at.desc()).first()
            
            if embedding:
                vec = pickle.loads(embedding.vector)
                score = np.dot(vec, query_vec) / (np.linalg.norm(vec) * np.linalg.norm(query_vec))
                if score > 0.5:
                    results.append((chunk, score))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def from_video_search(self, query_vec, model_name="default", top_k=3):
        """Search for similar content from video embedding"""
        results = []
        
        chunks = self.session.query(Chunk).join(Embedding).filter(
            Embedding.model_name == model_name
        ).all()
        
        for chunk in chunks:
            embedding = self.session.query(Embedding).filter(
                Embedding.chunk_id == chunk.id,
                Embedding.model_name == model_name
            ).order_by(Embedding.created_at.desc()).first()
            
            if embedding:
                vec = pickle.loads(embedding.vector)
                vec = vec.flatten()
                query_vec = query_vec.flatten()
                score = np.dot(vec, query_vec) / (np.linalg.norm(vec) * np.linalg.norm(query_vec))
                print(f"score:{score}")
                if score > 0.1:
                    results.append(chunk.text)
                    
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def embedding_model_choose(self, model):
        """Choose embedding model"""
        if model == "ali-text-embedding-v3":
            return get_embedding_ali
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
    
    def AI_Analyse(self, prompt, embedding_model="languagebind_text", chat_model="gemini-3-pro-preview"):
        """Call API for analysis"""
        context_chunks = [chunk.text for chunk, score in search_results]
        context = "\n\n---\n\n".join(context_chunks)

        print("Context found, generating response...")
        
        try:
            response = self.chat_model_choose(chat_model)(prompt=prompt, context=context)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return

        print("\nModel response:")
        print(response)


# if __name__ == "__main__":
#     txt_folder_path = os.path.join(project_root, "RAG", "txt_files") 
#     target_db_folder = os.path.join(project_root, "RAG", "db_files") 
#     db_filename = "LLM_Knowledge_En.db"
#     txt_folder_path = r"D:\pythonWorks\SpatialTemporalAttentionGCN-master\SpatialTemporalAttentionGCN-master\whole_dataset_txt!!!!!!!!\txt"
#     api_key = "dab1885a-b079-41cc-af7a-e15b3694fa87:fx"
#     translator = DeeplTranslator(api_key=api_key, source="zh", target="en", use_free_api=True)
#     # print(f"Text source folder: {txt_folder_path}")
#     # print(f"Target DB folder: {target_db_folder}")

#     # if not os.path.exists(target_db_folder):
#     #     os.makedirs(target_db_folder)
#     #     print(f"Created folder: {target_db_folder}")
    
#     # new_db_path = os.path.join(target_db_folder, db_filename)
#     # new_db_url = f"sqlite:///{new_db_path}"
    
#     # print(f"Creating/Connecting to database at: {new_db_path}")
    
#     # new_engine = create_engine(new_db_url, echo=False)
#     # NewSession = sessionmaker(bind=new_engine)
#     # new_session = NewSession()
    

#     # Base.metadata.create_all(new_engine)
    

#     new_db_manager = KnowledgeDB(session)
#     # new_db_manager.clear_chunks()
#     # new_db_manager.scan_and_add_chunks(chunk_size=250)

#     # if os.path.exists(txt_folder_path):
#     #     txt_files = [f for f in os.listdir(txt_folder_path) if f.endswith('.txt')]
#     #     print(f"Found {len(txt_files)} txt files.")
        
#     #     for filename in txt_files:
#     #         file_path = os.path.join(txt_folder_path, filename)
#     #         try:
#     #             with open(file_path, 'r', encoding='utf-8') as f:
#     #                 content = f.read()
                    

#     #             merged_content = content.replace('\n', '').replace('\r', '')
#     #             translated_text = translator.translate(merged_content)
#     #             print(translated_text)
#     #             translated_text = translated_text.replace('\n', '').replace('\r', '')
#     #             print(f"Processing: {filename}")
#     #             new_db_manager.add_document(title=filename, content=translated_text)
                
#     #         except Exception as e:
#     #             print(f"Failed to process {filename}: {e}")
                

#     print("Updating embeddings...")
#     try:
#         new_db_manager.update_embeddings(
#             embed_fn=get_embedding_ali, 
#             model_name="ali-text-embedding-v3"
#         )
#         print("Embedding update complete.")
#     except Exception as e:
#         print(f"Error updating embeddings: {e}")
                
#     # else:
#     #     print(f"Warning: Text folder not found at {txt_folder_path}")
    

#     # if __name__ == "__main__":
#     #     batch_translate_files(FOLDER_PATH, MY_API_KEY)


