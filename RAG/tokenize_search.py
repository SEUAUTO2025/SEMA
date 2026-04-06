from RAG.Knowledge_Database.RAGFunc import *
from RAG.Knowledge_Database.AIdbconfig import session, session_en
from RAG.Knowledge_Database.AI_dbmanager import KnowledgeDB

def Tokenize_SearchKeyword(
    video_path,
    pose_csv_path=None,
    pipeline=1,
    subpipeline=4,
    language='zh',
    show=False,
    progress_callback=None,
    return_keyword_polarity=False,
    assessment_model_name="qwen3-vl-plus",
):
    """
    Extract keywords from video and search knowledge database
    
    Args:
        video_path: Path to video file
        pose_csv_path: Optional pose CSV path for keyframe/biomechanics branches
        pipeline: 1 for text-based embedding, 2 for video-based embedding
        language: 'zh' for Chinese, 'en' for English
        assessment_model_name: assessment LLM model id used by get_video_ori_keywords(...)
    
    Returns:
        When `pipeline == 1`, returns `(score_dict, comment, retrieved_snippets)`.
        If `return_keyword_polarity=True`, appends a fourth item:
        `keyword_polarity = {"positive": [...], "negative": [...], "unknown": [...]}`.
    """
    if pipeline == 1:
        answer_content = get_video_ori_keywords(
            video_path,
            pose_csv_path=pose_csv_path,
            pipeline=subpipeline,
            model_name=assessment_model_name,
            language=language,
            show=show,
            progress_callback=progress_callback,
            target_k = 6
        )
        total_score = answer_content["total_score"]
        head_score  = answer_content["head_score"]
        hand_score  = answer_content["hand_score"]
        torso_score = answer_content["torso_score"]
        foot_score  = answer_content["foot_score"]
        arm_score   = answer_content["arm_score"]
        score_dict = {
            "total_score": total_score,
            "head_score": head_score,
            "hand_score": hand_score,
            "torso_score": torso_score,
            "foot_score": foot_score,
            "arm_score": arm_score
        }
        comment = answer_content["comment"]
        keyword_polarity = answer_content.get(
            "comment_polarity",
            {"positive": [], "negative": [], "unknown": []},
        )
        if isinstance(comment, (list, tuple)):
            query_texts = [str(item).strip() for item in comment if str(item).strip()]
        else:
            query_text = str(comment).strip()
            query_texts = [query_text] if query_text else []
        if not query_texts:
            query_texts = [str(score_dict)]
        query_embeddings = get_embedding(query_texts)
        db = KnowledgeDB(session=session_en if language == 'en' else session)
        top_k = 34 if language == 'en' else 34
        if callable(progress_callback):
            try:
                progress_callback("retrieving_knowledge", "正在检索技术知识库")
            except Exception:
                pass
        retrieved = db.from_video_search(query_vec=query_embeddings, model_name='ali-text-embedding-v3', top_k=top_k)
        if return_keyword_polarity:
            return score_dict, comment, retrieved, keyword_polarity
        return score_dict, comment, retrieved
    
    elif pipeline == 2:
        video_token = get_embedding_languagebind_video(video_path)
        db = KnowledgeDB(session=session_en if language == 'en' else session)
        if language == 'en':
            return db.from_video_search(query_vec=video_token, model_name='languagebind', top_k=34)
        else:
            return db.from_video_search(query_vec=video_token, model_name='languagebind', top_k=17)
