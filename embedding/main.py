import onnxruntime
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, PretrainedConfig
import os

# 设置 FastAPI
app = FastAPI()

# 定义请求体模型
class TextRequest(BaseModel):
    input: list[str]
    

# 定义返回的模型
class EmbeddingResponse(BaseModel):
    data: list[dict]

# Mean pool function
def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

# 加载 tokenizer 和模型配置
tokenizer = AutoTokenizer.from_pretrained('./models/jina-embeddings-v3')
config = PretrainedConfig.from_pretrained('./models/jina-embeddings-v3')

# 加载 ONNX 模型
model_path = os.path.abspath('./models/jina-embeddings-v3/model.onnx')
session = onnxruntime.InferenceSession(model_path)

# 嵌入计算函数
def compute_embedding(input: str):
    # Tokenize 输入文本
    input_text = tokenizer(input, return_tensors='np', padding=True, truncation=True)
    
    # 准备输入 ONNX 模型
    task_type = 'text-matching'  # 假设使用 `text-matching` 任务
    task_id = np.array(config.lora_adaptations.index(task_type), dtype=np.int64)
    inputs = {
        'input_ids': input_text['input_ids'],
        'attention_mask': input_text['attention_mask'],
        'task_id': task_id
    }
    # 执行推理
    outputs = session.run(None, inputs)[0]
    # 应用 mean pooling 和归一化
    embeddings = mean_pooling(outputs, input_text["attention_mask"])
    embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)  # 归一化
    return embeddings

# 路由：获取文本嵌入
@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embedding(request: TextRequest):
    input = request.input
    embeddings = compute_embedding(input)
    data = []
    for idx,emb in enumerate(embeddings):
        data.append({"index":idx,"embedding":emb.tolist()})
    # 返回嵌入结果
    return EmbeddingResponse(data=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6666)