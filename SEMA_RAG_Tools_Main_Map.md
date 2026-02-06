# SEMA 代码梳理（RAG / Tools/Gen_dataset / main 评估逻辑）

## 范围说明
- 已覆盖：`RAG/`（排除 `LanguageBind_main`）、`Tools/Gen_dataset/`、`main.py` 里“系统生成评估文本”的相关逻辑
- 数据库文件：`RAG/db_files/*.db` 未解析结构，仅记录用途

---

## RAG 目录

### 1) `RAG/tokenize_search.py`
**核心函数：`Tokenize_SearchKeyword(video_path, pipeline=1, language='zh')`**
- 作用：从视频提取关键词或视频嵌入，并从知识库中检索相似文本片段
- 流程：
  - `pipeline=1`（文本流）
    - `get_video_ori_keywords_ali(video_path)` 生成关键词文本
    - `get_embedding_ali([answer_content])` 生成文本向量
    - `KnowledgeDB.from_video_search(...)` 检索
  - `pipeline=2`（视频流）
    - `get_embedding_languagebind_video(video_path)` 生成视频向量
    - `KnowledgeDB.from_video_search(...)` 检索
- 检索 top_k：
  - 英文 `language='en'` 用 `top_k=34`
  - 中文 `top_k=17`
- 依赖：
  - `RAG.Knowledge_Database.RAGFunc`
  - `RAG.Knowledge_Database.AIdbconfig`
  - `RAG.Knowledge_Database.AI_dbmanager.KnowledgeDB`

---

### 2) `RAG/Knowledge_Database/AIdbconfig.py`
**用途：数据库连接配置**
- 计算 `project_root = RAG/Knowledge_Database` 的父目录
- 数据库路径固定为：
  - `RAG/db_files/LLM_Knowledge_New_ali_zh.db`
- 暴露：
  - `engine`
  - `session`（SQLAlchemy session）

---

### 3) `RAG/Knowledge_Database/AI_dbmanager.py`
**用途：数据库表定义 + 业务操作**

**数据表模型**
- `Document`: `id`, `title`, `content`
- `Chunk`: `id`, `document_id`, `text`, `embedding`
- `Embedding`: `id`, `chunk_id`, `vector`, `model_name`, `created_at`

**关键方法**
- `add_document(title, content, embed_fn=None, model_name="default", chunk_size=70)`
  - 保存 Document
  - `split_and_merge` 生成 Chunk
- `clear_chunks()`
  - 删除 `Embedding` + `Chunk`
- `scan_and_add_chunks(chunk_size=70)`
  - 遍历现有 Document 重新切块
- `delete_document(document_id)`
  - 删除 Document + 相关 Chunk + Embedding
- `delete_embedding()`
  - 删除所有 Embedding
- `update_embeddings(embed_fn, model_name="default", batch_size=10)`
  - 批量为 chunks 生成向量并入库
- `clear_database()`
  - 清空所有表内容
- `search(query, embed_fn, model_name="default", top_k=3)`
  - 对文本 query 向量化后与 Embedding 相似度搜索
  - 余弦相似度阈值 `> 0.5`
- `from_video_search(query_vec, model_name="default", top_k=3)`
  - 用视频向量搜索 Embedding
  - 余弦相似度阈值 `> 0.1`
  - 返回 `chunk.text` 列表

**模型选择**
- `embedding_model_choose(model)`
  - `"ali-text-embedding-v3" -> get_embedding_ali`
  - `"languagebind_video" -> get_embedding_languagebind_video`
  - `"languagebind_text" -> get_embedding_languagebind_text`

**对话逻辑**
- `chat(user_question, top_k=3, embedding_model="ali-text-embedding-v3", chat_model="deepseek")`
  - 从数据库检索 -> 拼上下文 -> 调用 `chat_model_choose`
  - 注意：`chat_model_choose` 在当前文件中未定义

**潜在问题（需要后续确认）**
- `absolute_project_root` 写死为 `D:\Pythonworks\SpatialTemporalAttentionGCN-master\...`，与当前项目不一致
- `from_video_search` 最后 `results.sort(key=lambda x: x[1], reverse=True)` 但 `results` 是文本字符串，索引不合法
- `AI_Analyse` 中使用 `search_results` 变量未定义
- `chat_model_choose` 未在此文件定义，可能在别处或遗失

---

### 4) `RAG/Knowledge_Database/RAGFunc.py`
**用途：文本切块、嵌入、LLM 调用、LanguageBind 多模态**

**核心函数**
- `split_and_merge(text, chunk_size=500)`
  - 使用 `RecursiveCharacterTextSplitter`
  - 保持段落/句子完整，避免跨段
  - 合并到接近 `chunk_size`

- `get_embedding_ali(texts)`
  - 阿里通义兼容 OpenAI 接口
  - 模型：`text-embedding-v4`
  - 输出 `np.float32` 向量

- `get_video_ori_keywords_ali(video_path)`
  - 使用 `qwen3-vl-plus` 读取视频并输出关键词
  - 强约束提示词，输出仅关键词，无格式
  - `stream=True`，拼接所有 chunk 的 `delta.content`

- `get_response_ali(keywords, pipeline=1, math_feature=None)`
  - 用关键词生成“系统评估文本 + 评分”
  - 生成 JSON 字符串，字段：
    - `total_score`, `head_score`, `hand_score`, `torso_score`, `foot_score`, `arm_score`, `comment`
  - 模型：`qwen-plus`（通义兼容 OpenAI 接口）
  - `pipeline=2` 支持额外数值特征 `math_feature`，但当前项目里常以 `None` 调用

- `get_embedding_languagebind_text(texts)`
  - LanguageBind 文本编码
  - tokenizer 缓存路径 `./cache_dir/tokenizer_cache_dir`

- `get_embedding_languagebind_video(video_path)`
  - LanguageBind 视频编码
  - 依赖 `LanguageBind_Video_FT`

- `construct_complex_prompt(scores, prompt, comment_text)`
  - 拼接分数 JSON + 数据库检索文本，构造综合 prompt

---

## Tools/Gen_dataset 目录

### 1) `Tools/Gen_dataset/dataset_exe.py`
**用途：关键点字典 + CSV 数据加载**

**关键点配置**
- `WHOLEBODY3D_KEYPOINT_NAMES`
- `HEAD_KEYPOINT_NAMES`
- `HAND_KEYPOINT_NAMES`
- `FOOT_KEYPOINT_NAMES`
- `TORSO_KEYPOINT_NAMES`
- `ARM_KEYPOINT_NAMES`
- 对应骨架连接 `*_BONE_CONNECTIONS`
- 原始索引映射 `*_ORIGINAL_INDICES`

**核心函数**
- `load_single_csv_with_multipart_labels(csv_path, max_frames=MAX_FRAMES)`
  - 读取 CSV：
    - `frame,x0,y0,z0,...,x132,y132,z132,label_hand,label_head,label_feet,label_arm,label_body,label_total`
  - 输出：
    - `data` shape `(3, T, 133, 1)`
    - `labels` dict: `hand/head/feet/arm/body`
    - `label_total`

---

### 2) `Tools/Gen_dataset/model_config.py`
**用途：训练模型的部位配置映射**

- 复用 `dataset_exe.py` 的关键点字典和连接定义
- 统一 `model_configs` 映射：
  - `head_model`
  - `hand_model`
  - `foot_model`
  - `torso_model`
  - `arm_model`
- 每个配置包含：
  - `indices`, `keypoint_names`, `bone_connections`, `label_key`

---

## main.py 中“系统生成评估文本”逻辑

### 入口流程（`if __name__ == '__main__':`）
1. `Keypoint_Extract(video_path)`
2. `Extract_Bodypart_Data(...)`
3. `extract_action_features(math_feature)`
4. `keywords = Tokenize_SearchKeyword(video_path, pipeline=1, language='en')`
5. `answer = get_response_ali(keywords, pipeline=1, math_feature=None)`
6. `data = json.loads(answer)`
7. 组装 `scores` 字典
8. `txt_label, label, label_total = get_matching_text(video_path, txt_dir, csv_dir)`
9. 文本评估指标：
   - BLEU
   - BERTScore
   - CIDEr-D
   - METEOR
10. 打印评分与 GT 对比

---

### 关键点说明

**1) 关键词与评估文本生成**
- 关键词来自：
  - `get_video_ori_keywords_ali(video_path)`（Qwen3-VL 从视频生成关键词）
  - 再嵌入 + DB 检索（`KnowledgeDB.from_video_search`）
- 评估文本来自：
  - `get_response_ali`（Qwen-Plus 生成 JSON 文本）
  - JSON 里 `comment` 为实际评估文本

**2) 地面真值文本与分数**
- `get_matching_text(video_path, txt_dir, csv_dir)`：
  - 查找同名 `.txt` 与 `.csv`
  - `.txt` 返回为 GT 评估文本
  - `.csv` 返回标签分数 `label` 和 `label_total`

**3) 评价指标**
- BLEU 使用 NLTK
- BERTScore 使用 `bert_score.score`
- CIDEr 使用 `pycocoevalcap`
- METEOR 使用 NLTK

---

## 注意事项（后续排查建议）
- `get_response_ali` 直接 `json.loads`，若模型输出不严格 JSON 会崩
- `math_feature_input` 在 `main.py` 中未被使用
- `Tokenize_SearchKeyword` pipeline=1 的返回依赖数据库中已存在 embeddings
- 数据库路径固定为 `LLM_Knowledge_New_ali_zh.db`，更换语言或 DB 需改配置

---

## 文件索引（便于快速定位）
- `RAG/tokenize_search.py`
- `RAG/Knowledge_Database/AIdbconfig.py`
- `RAG/Knowledge_Database/AI_dbmanager.py`
- `RAG/Knowledge_Database/RAGFunc.py`
- `Tools/Gen_dataset/dataset_exe.py`
- `Tools/Gen_dataset/model_config.py`
- `main.py`
