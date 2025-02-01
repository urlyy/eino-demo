package main

import (
	"context"
	"log"
	"os"
	"strconv"

	file "github.com/cloudwego/eino-ext/components/document/loader/file"
	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/markdown"
	"github.com/cloudwego/eino/components/document"
	"github.com/joho/godotenv"
	"github.com/pinecone-io/go-pinecone/v2/pinecone"
	"github.com/urlyy/eino-demo/hello_rag/components/myembedder"
	"github.com/urlyy/eino-demo/hello_rag/utils"
	"google.golang.org/protobuf/types/known/structpb"
)

func main() {
	_ = godotenv.Load()
	// 创建并配置 ChatModel
	pineconeAPIKey := os.Getenv("PINECONE_APIKEY")
	pineconeHost := os.Getenv("PINECONE_HOST")
	ctx := context.Background()
	loader, err := file.NewFileLoader(ctx, &file.FileLoaderConfig{
		UseNameAsID: true,
	})
	if err != nil {
		panic(err)
	}
	// len(docs)==1, docs[0]={ID:说明书.md, Content:...}
	docs, err := loader.Load(ctx, document.Source{
		URI: "./说明书.md",
	})
	if err != nil {
		panic(err)
	}
	// 初始化分割器
	splitter, err := markdown.NewHeaderSplitter(ctx, &markdown.HeaderConfig{
		Headers: map[string]string{
			"#":   "h1",
			"##":  "h2",
			"###": "h3",
		},
		TrimHeaders: false,
	})
	if err != nil {
		panic(err)
	}
	chunks, err := splitter.Transform(ctx, docs)
	if err != nil {
		panic(err)
	}
	idxConn, err := utils.CreatePineconeConn(pineconeAPIKey, pineconeHost)
	if err != nil {
		panic(err)
	}
	embedder, err := myembedder.NewEmbedder(ctx)
	if err != nil {
		panic(err)
	}
	contents := make([]string, len(chunks))
	for i, chunk := range chunks {
		contents[i] = chunk.Content
	}
	embedResults, err := embedder.EmbedStrings(ctx, contents)
	if err != nil {
		panic(err)
	}
	vectors := make([]*pinecone.Vector, len(chunks))
	startID := 0
	for i, chunk := range chunks {
		metadata, err := structpb.NewStruct(map[string]interface{}{
			"content": chunk.Content,
		})
		if err != nil {
			log.Fatalf("Failed to create metadata map: %v", err)
		}
		vectors[i] = &pinecone.Vector{
			Id:       strconv.Itoa(startID + i),
			Values:   utils.SliceFloat64To32(embedResults[i]),
			Metadata: metadata,
		}
	}
	count, err := idxConn.UpsertVectors(ctx, vectors)
	if err != nil {
		panic(err)
	}
	log.Printf("Successfully upserted %d vector(s)!\n", count)
}
