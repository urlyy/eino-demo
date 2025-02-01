package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
)

var agent compose.Runnable[map[string]any, []*schema.Message]

const systemPrompt = "你是一个SQL专家."
const userTemplate = "商品搜索。product表的字段有id、name、color、price、comment_num、is_deleted。只能选择未删除的商品即is_deleted=false的。分页查询,每页10个,每次只搜第一页。"

var llmTemp float32 = 0.7

type SearchProductParams struct {
	SQL string `json:"sql" jsonschema:"description=SQL for searching products."`
}

type SearchProductsResponse struct {
	Success  bool             `json:"success"`
	Message  string           `json:"message"`
	Products []map[string]any `json:"products"`
}

func SearchProductFunc(ctx context.Context, params *SearchProductParams) (SearchProductsResponse, error) {
	fmt.Printf("1. LLM已生成sql,传入了SearchProductFunc并准备与数据库交互: %+v\n", *params)
	// 具体的调用逻辑
	// 如：gorm.find(params.SQL, &products)
	products := []map[string]any{
		{"id": 1, "name": "手机壳1", "color": "红色", "price": 50, "comment_num": 120, "is_deleted": false},
		{"id": 2, "name": "2手机壳", "color": "红色", "price": 40, "comment_num": 130, "is_deleted": false},
	}
	resp := SearchProductsResponse{
		Success:  true,
		Message:  "查询成功",
		Products: products,
	}
	return resp, nil
}

func init() {
	ctx := context.Background()
	searchProductTool, err := utils.InferTool("search_product", "基于用户需求进行商品搜索", SearchProductFunc)
	if err != nil {
		log.Fatal(err)
	}
	// 初始化 tools
	tools := []tool.BaseTool{
		searchProductTool,
	}
	// 获取工具信息, 用于绑定到 ChatModel
	toolInfos := make([]*schema.ToolInfo, 0, len(tools))
	for _, tool := range tools {
		info, err := tool.Info(ctx)
		if err != nil {
			log.Fatal(err)
		}
		toolInfos = append(toolInfos, info)
	}
	// 获取环境变量
	err = godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	// 创建并配置 ChatModel
	openAIAPIKey := os.Getenv("OPENAI_API_KEY")
	openAIBaseURL := os.Getenv("OPENAI_BASE_URL")
	openAIModelName := os.Getenv("OPENAI_MODEL_NAME")
	chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		BaseURL:     openAIBaseURL,
		Model:       openAIModelName,
		APIKey:      openAIAPIKey,
		Temperature: &llmTemp,
	})
	if err != nil {
		log.Fatal(err)
	}
	// 将 tools 绑定到 ChatModel
	err = chatModel.BindTools(toolInfos)
	if err != nil {
		log.Fatal(err)
	}
	// 创建 tools 节点
	toolsNode, err := compose.NewToolNode(ctx, &compose.ToolsNodeConfig{
		Tools: tools,
	})
	if err != nil {
		log.Fatal(err)
	}
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
	// 构建完整的处理链
	chain := compose.NewChain[map[string]any, []*schema.Message]()
	chain.
		AppendChatTemplate(template).
		AppendChatModel(chatModel).
		AppendToolsNode(toolsNode)

	// 编译 chain
	agent, err = chain.Compile(ctx)
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	userInput := "搜索评论数在20到30之间、价格小于等于50的红色手机壳"
	userPrompt := fmt.Sprintf("%s%s", userTemplate, userInput)
	variables := map[string]any{
		"task": userPrompt,
	}
	resp, err := agent.Invoke(context.Background(), variables)
	if err != nil {
		log.Fatal(err)
	}
	// 输出结果
	respString := resp[0].Content
	var respData SearchProductsResponse
	err = json.Unmarshal([]byte(respString), &respData)
	if err != nil {
		fmt.Printf("Error unmarshalling JSON: %+v\n", err)
		return
	}
	fmt.Printf("2. products: %+v\n", respData.Products)
}
