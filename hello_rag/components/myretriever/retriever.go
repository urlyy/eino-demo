package myretriever

import (
	"context"
	"encoding/json"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/pinecone-io/go-pinecone/v2/pinecone"
	"github.com/urlyy/eino-demo/hello_rag/components/myembedder"
	"github.com/urlyy/eino-demo/hello_rag/utils"
)

type Retriever struct {
	idxConn  *pinecone.IndexConnection
	embedder *myembedder.Embedder
}

func NewRetriever(ctx context.Context, apikey string, host string) (*Retriever, error) {
	idxConn, err := utils.CreatePineconeConn(apikey, host)
	if err != nil {
		return nil, err
	}
	embedder, err := myembedder.NewEmbedder(ctx)
	if err != nil {
		return nil, err
	}
	return &Retriever{
		idxConn:  idxConn,
		embedder: embedder,
	}, nil
}

var defaultTopK = 3

func prettifyStruct(obj interface{}) string {
	bytes, _ := json.MarshalIndent(obj, "", "  ")
	return string(bytes)
}
func (r *Retriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	queryVectors, err := r.embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	queryVector := utils.SliceFloat64To32(queryVectors[0])
	// 这是默认配置
	defaultOptions := &retriever.Options{
		TopK: &defaultTopK,
	}
	// 用传入的覆盖默认的
	options := retriever.GetCommonOptions(defaultOptions, opts...)
	res, err := r.idxConn.QueryByVectorValues(ctx, &pinecone.QueryByVectorValuesRequest{
		Vector: queryVector,
		TopK:   uint32(*options.TopK),
		// 不需要返回向量
		IncludeValues:   false,
		IncludeMetadata: true,
	})
	if err != nil {
		return nil, err
	}
	docs := make([]*schema.Document, len(res.Matches))
	for i, match := range res.Matches {
		metadata := match.Vector.Metadata.AsMap()
		docs[i] = &schema.Document{
			ID:      match.Vector.Id,
			Content: metadata["content"].(string),
		}
	}
	return docs, nil
}
