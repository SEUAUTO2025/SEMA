import sys
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from RAG.Knowledge_Database.RAGFunc import *
from RAG.Knowledge_Database.AIdbconfig import session
from RAG.Knowledge_Database.AI_dbmanager import KnowledgeDB

def Tokenize_SearchKeyword(video_path, pipeline=1, language='zh'):
    """
    Extract keywords from video and search knowledge database
    
    Args:
        video_path: Path to video file
        pipeline: 1 for text-based embedding, 2 for video-based embedding
        language: 'zh' for Chinese, 'en' for English
    
    Returns:
        List of keywords from knowledge database
    """
    if pipeline == 1:
        answer_content = get_video_ori_keywords_ali(video_path)
        query_embeddings = get_embedding_ali([answer_content])
        db = KnowledgeDB(session=session)
        if language == 'en':
            return db.from_video_search(query_vec=query_embeddings[0], model_name='ali-text-embedding-v3', top_k=34)
        else:
            return db.from_video_search(query_vec=query_embeddings[0], model_name='ali-text-embedding-v3', top_k=17)
    
    elif pipeline == 2:
        video_token = get_embedding_languagebind_video(video_path)
        db = KnowledgeDB(session=session)
        if language == 'en':
            return db.from_video_search(query_vec=video_token, model_name='languagebind', top_k=34)
        else:
            return db.from_video_search(query_vec=video_token, model_name='languagebind', top_k=17)