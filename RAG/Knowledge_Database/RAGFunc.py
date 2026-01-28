"""
@filename: RAG.py
@description: Implementation of all RAG functions
"""
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from openai import OpenAI
import requests
import json
from LanguageBind_main.languagebind import LanguageBindImageTokenizer,LanguageBind, to_device, LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor, transform_dict
import torch
from google import genai
import json
from typing import List, Union, Dict
import base64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_and_merge(text, chunk_size=500):
    """
    Split text using LangChain, then merge into chunks close to chunk_size
    - Does not cross paragraphs
    - Does not break sentences
    - Punctuation at end goes to previous chunk
    - Remove leading punctuation from chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";"]
    )
    pieces = splitter.split_text(text)

    chunks = []
    current_chunk = ""
    current_len = 0
    punctuations = {"。", ".", "！", "!", "？", "?", "；", ";"}

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if "\n\n" in piece:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_len = 0
            piece = piece.replace("\n\n", "").strip()
            if not piece:
                continue

        if piece in punctuations:
            current_chunk += piece
            current_len += len(piece)
            continue

        piece_len = len(piece)
        if current_len + piece_len <= chunk_size:
            current_chunk += piece
            current_len += piece_len
        else:
            if current_chunk:
                while current_chunk and current_chunk[0] in punctuations:
                    current_chunk = current_chunk[1:].lstrip()
                if current_chunk:
                    chunks.append(current_chunk.strip())
            current_chunk = piece
            current_len = piece_len

    if current_chunk:
        while current_chunk and current_chunk[0] in punctuations:
            current_chunk = current_chunk[1:].lstrip()
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

def get_embedding_ali(texts):
    """Call OpenAI API to get text embedding vectors"""
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions=1024,
        encoding_format="float"
    )
    embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    return embeddings


def video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        encoded = base64.b64encode(video_file.read()).decode("utf-8")
    return encoded

def get_video_ori_keywords_ali(video_path):
    """Extract keywords from video using Alibaba Cloud API"""
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    enable_thinking = False

    video_base64 = video_to_base64(video_path)
    
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        },
                    },
                    {"type": "text", "text": "You are a professional archery coach. Please carefully review the video content and provide a preliminary assessment and improvement suggestions for the archery posture from five aspects: head, hands, shoulders, torso, and feet. Output only a concise summary containing keywords from the assessment and improvement suggestions (content should not be too long). Do not output any other content."},
                    {"type": "text", "text": "When replying, please include the following keywords: 1. Correct actions: body axis perpendicular to ground, feet shoulder-width apart, body sideways to shooting direction, front hand/front shoulder/rear shoulder aligned, front forearm internally rotated, front hand fingers relaxed, rear forearm aligned with arrow, rear hand against chin, head rotates horizontally, rear hand doesn't slip during aiming, rear hand extends backward during release, front hand relaxes during release. 2. Incorrect actions: body center off-balance, feet positioned randomly, body not turned sideways, front shoulder angled/not extended, forearm externally rotated, fingers stiff or clenched, forearm angled to arrow, rear hand suspended/not planted, string not touching chin and nose, head rotation incomplete."},
                    {"type": "text", "text": "Output should only contain the above keywords and punctuation, no other information or formatting, no bullet points or numbering, as concise as possible."},
                ],
            }
        ],
        stream=True,
    )

    if enable_thinking:
        print("\n" + "=" * 20 + "Thinking Process" + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "Complete Response" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end='', flush=True)
                answer_content += delta.content
                
    return answer_content

def get_response_ali(keywords: list[str], pipeline=1, math_feature=None) -> str:
    """
    Generate assessment text based on keyword list
    :param keywords: List of keywords
    :param pipeline: Pipeline mode (1 or 2)
    :param math_feature: Biomechanical features (optional)
    :return: Model-generated assessment text
    """
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    if pipeline == 1:
        system_instruction = (
            "You are a professional archery coach. Your task is to generate archery assessment text and scores (0-5) for five body parts: head, hands, torso, feet, and arms, plus total score (0-25).\n"
            "Rules:\n"
            "1. External searches or references beyond provided information are strictly prohibited.\n"
            "2. Only perform logical connections and analysis based on provided keywords, use original retrieved terms as much as possible.\n"
            "3. Output should include: action quality assessment and improvement suggestions.\n"
            "4. Ensure text is coherent, organized, and professional.\n"
            "5. Standard format: This student's correct actions are:... This student's incorrect actions are:...\n"
            "6. Ensure no contradictory statements before output.\n"
            "7. Output JSON format string with keys: 'total_score', 'head_score', 'hand_score', 'torso_score', 'foot_score', 'arm_score', 'comment'.\n"
            "8. Scoring criteria: Check if keywords contain descriptions of correct/incorrect actions for each body part. More correct actions and fewer incorrect actions = higher score. Scores must be integers.\n"
            "9. Keep response around 550 characters to align with actual evaluation length."
        )
    elif pipeline == 2:
        system_instruction = (
            "You are a professional archery coach. Your task is to generate archery assessment text and scores (0-5) for five body parts: head, hands, torso, feet, and arms, plus total score (0-25).\n"
            "Rules:\n"
            "1. External searches or references beyond provided information are strictly prohibited.\n"
            "2. Only perform logical connections and analysis based on provided keywords, use original retrieved terms as much as possible.\n"
            "3. Output should include: action quality assessment and improvement suggestions.\n"
            "4. Ensure text is coherent, organized, and professional.\n"
            "5. Standard format: This student's correct actions are:... This student's incorrect actions are:...\n"
            "6. Ensure no contradictory statements before output.\n"
            "7. Output JSON format string with keys: 'total_score', 'head_score', 'hand_score', 'torso_score', 'foot_score', 'arm_score', 'comment'.\n"
            "8. Scoring criteria: Check if keywords contain descriptions of correct/incorrect actions for each body part. More correct actions and fewer incorrect actions = higher score. Scores must be integers.\n"
            "9. Keep response around 550 characters to align with actual evaluation length.\n"
            f"10. When generating assessment and scoring, also refer to these metrics to adjust conclusions:\n"
            f"(1) Hand-shoulder-hand angle (closer to 180° = more standard arm movement. ≥160° = excellent, arm score 4-5, positive evaluation; 152.7-160° = good, score 3-4; 142.7-152.7° = average, score 2-3, may have negative comments; <142.7° = poor, score 1-2, must have negative comments): {math_feature['max_angle_avg']}\n"
            f"(2) Hand-to-chin distance (smaller = more standard draw. ≤0.1026 = excellent, hand score 4-5, positive evaluation; 0.1026-0.1092 = good, score 3-4; 0.1092-0.1150 = average, score 2-3, may have negative comments; >0.1150 = poor, score 1-2, must have negative comments): {math_feature['min_dist_avg']}\n"
            f"(3) Body center alignment (smaller = more vertical. ≤0.001 = excellent, torso score 4-5, positive evaluation; 0.001-0.002 = good, score 3-4; 0.002-0.0027 = average, score 2-3, may have negative comments; >0.0027 = poor, score 1-2, must have negative comments): {math_feature['min_x_diff_avg']}"
        )

    keyword_str = "\n".join([f"- {k}" for k in keywords])
    user_prompt = f"Please generate an evaluation text based on the following keywords (provide a single paragraph response without special characters):\n\n{keyword_str}"

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]

    answer_content = ""
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", 
            messages=messages,
            stream=True,
            temperature=0.7,
        )

        print("\n" + "=" * 20 + " Generating Assessment Text " + "=" * 20)
        is_answering = False
        
        for chunk in completion:
            delta = chunk.choices[0].delta
            
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if not is_answering:
                    print(f"\n[Thinking]: {delta.reasoning_content}", end="", flush=True)

            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "-" * 45)
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

        print("\n" + "=" * 45)
        
    except Exception as e:
        print(f"API call error: {e}")
        return f"Error: {e}"

    return answer_content

def get_embedding_languagebind_text(texts):
    """Get text embeddings using LanguageBind model"""
    texts = texts if isinstance(texts, list) else [texts]
    device = torch.device("cuda:0")
    
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        'lb203/LanguageBind_Image',
        cache_dir='./cache_dir/tokenizer_cache_dir'
    )
    
    model = LanguageBind(clip_type={'image': 'LanguageBind_Image'},
                         cache_dir='./cache_dir').to(device)
    model.eval()
    
    inputs = {
        'language': to_device(tokenizer(texts, max_length=77, 
                                        padding='max_length', 
                                        truncation=True, 
                                        return_tensors='pt'), 
                                        device)
    }
    
    with torch.no_grad():
        embeddings = model(inputs)['language']
    return embeddings.cpu().numpy()

def get_embedding_languagebind_video(video_path):
    """Get video embeddings using LanguageBind model"""
    video_path = video_path if isinstance(video_path, list) else [video_path]
    clip_type = {
        'video': 'LanguageBind_Video_FT',
    }
    model = LanguageBind(clip_type=clip_type)
    model = model.to(device)
    model.eval()
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    inputs = {
        'video': to_device(modality_transform['video'](video_path), device)
    }
    
    with torch.no_grad():
        embeddings = model(inputs)['video']
    return embeddings.cpu().numpy()

def construct_complex_prompt(scores: Dict, prompt: str, comment_text: List[str]):
    """
    Construct a complex prompt containing structured scores, retrieved comments, and user instructions
    """
    scores_str = json.dumps(scores, indent=2, ensure_ascii=False)

    formatted_comments = "\n".join([f"{i+1}. {text}" for i, text in enumerate(comment_text)])
    print(formatted_comments)

    final_content = f"""Please evaluate the archer's posture based on the following information:
    1. Movement Scores for Each Body Part of Athletes (json):
    {scores_str}
    2. Evaluation keywords and phrases provided by professional coaches retrieved from the database:
    {formatted_comments}
    """
    return final_content
