package myembedder

import (
	"context"

	"github.com/cloudwego/eino-ext/components/embedding/openai"
	"github.com/cloudwego/eino/components/embedding"
)

var (
	defaultDim = 1024
)

type Embedder struct {
	openaiEmbedder *openai.Embedder
}

func NewEmbedder(ctx context.Context) (*Embedder, error) {
	config := openai.EmbeddingConfig{
		BaseURL:    "http://localhost:6666",
		Dimensions: &defaultDim,
		Timeout:    0,
	}
	openaiEmbedder, err := openai.NewEmbedder(ctx, &config)
	return &Embedder{
		openaiEmbedder: openaiEmbedder,
	}, err
}

func (e *Embedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	return e.openaiEmbedder.EmbedStrings(ctx, texts)
}
