一个使用Eino实现的集成知识库的LLM chat demo

测试方式：
1. 先申请免费的pinecone serverless
2. 启动 ../embedding/main.py
3. 运行 add_index_data/main.go ,将`说明书.md`的内容向量化并存在pinecone中
4. 运行 chat/main.go，查看输出结果