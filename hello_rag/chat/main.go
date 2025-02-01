package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
	"github.com/urlyy/eino-demo/hello_rag/components/myretriever"
)

const systemPrompt = "你是一个客服"

var llmTemp float32 = 0.7

func NewChain() (compose.Runnable[map[string]any, *schema.Message], error) {
	ctx := context.Background()
	// 获取环境变量
	_ = godotenv.Load()
	// 创建并配置 ChatModel
	openAIAPIKey := os.Getenv("OPENAI_API_KEY")
	openAIBaseURL := os.Getenv("OPENAI_BASE_URL")
	openAIModelName := os.Getenv("OPENAI_MODEL_NAME")
	chatModel, _ := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		BaseURL:     openAIBaseURL,
		Model:       openAIModelName,
		APIKey:      openAIAPIKey,
		Temperature: &llmTemp,
	})
	// 创建提示词模版节点
	template := prompt.FromMessages(schema.FString,
		&schema.Message{
			Role:    schema.System,
			Content: systemPrompt,
		},
		&schema.Message{
			Role:    schema.User,
			Content: "{task}。",
		},
	)
	chain := compose.NewChain[map[string]any, *schema.Message]()
	// === 1. 没有RAG ===
	// chain.
	// 	AppendChatTemplate(template, compose.WithNodeName("template")).
	// 	AppendChatModel(chatModel, compose.WithNodeName("chat_model"))
	// === 1. 没有RAG ===
	// === 2. 有RAG ===
	pineconeAPIKey := os.Getenv("PINECONE_APIKEY")
	pineconeHost := os.Getenv("PINECONE_HOST")
	retriever_, _ := myretriever.NewRetriever(ctx, pineconeAPIKey, pineconeHost)
	// 构建完整的处理链
	parallel := compose.NewParallel()
	ragChain := compose.NewChain[[]*schema.Message, string]()
	ragChain.
		AppendLambda(compose.InvokableLambda(func(_ context.Context, input []*schema.Message) (string, error) {
			fmt.Printf("initial msg is %+v\n", input)
			// 本demo不考虑history,故直接第2个msg
			if len(input) < 2 {
				return "", errors.New("input msg is not enough")
			}
			return input[1].Content, nil
		})).
		AppendRetriever(retriever_).
		AppendLambda(compose.InvokableLambda(func(_ context.Context, docs []*schema.Document) (string, error) {
			var contents []string
			for _, doc := range docs {
				contents = append(contents, doc.Content)
			}
			knowledge := strings.Join(contents, "\n")
			return knowledge, nil
		}))
	parallel.
		AddGraph("rag_chain", ragChain).
		AddPassthrough("pass_msg")
	ragReplacer := compose.InvokableLambda(func(_ context.Context, inputs map[string]any) ([]*schema.Message, error) {
		newPrompt := inputs["rag_chain"].(string)
		msgs := inputs["pass_msg"].([]*schema.Message)
		// 把之前的用户输入和刚匹配到的相关知识合并
		msgs[1].Content = fmt.Sprintf("user query: %s\nContext knowledge: %s", msgs[1].Content, newPrompt)
		// 优化大模型输出
		msgs[1].Content += "\n请仅根据上述上下文回答问题，不要提及来源或上下文本身。"
		// fmt.Printf("new msg is %+v\n", msgs)
		return msgs, nil
	})
	chain.
		AppendChatTemplate(template).
		AppendParallel(parallel).
		AppendLambda(ragReplacer).
		AppendChatModel(chatModel)
	// === 2. 有RAG ===
	// 编译 chain
	myChain, _ := chain.Compile(ctx)
	return myChain, nil
}

func main() {
	userInput := "我需要联系OmniPort任意门制造公司的客服，他们的联系电话是多少？"
	variables := map[string]any{
		"task": userInput,
	}
	opts := []compose.Option{
		compose.WithRetrieverOption(retriever.WithTopK(5)),
	}
	myChain, _ := NewChain()
	resp, _ := myChain.Invoke(context.Background(), variables, opts...)
	// 输出结果
	respString := resp.Content
	fmt.Println(respString)
}
