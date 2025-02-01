https://huggingface.co/jinaai/jina-embeddings-v3/tree/main
```
models
└─jina-embeddings-v3
    config.json
    model.onnx
    model.onnx_data
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
```

# Notice
考虑到部分文件过大，因此并未上传到仓库中，请自行在 https://huggingface.co/jinaai/jina-embeddings-v3/tree/main/onnx 目录下下载
- model.onnx_data

# 运行
```shell
pip install -r requirements.txt
cd embedding
python main.py
```