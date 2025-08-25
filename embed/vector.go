package embed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type OllamaResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// OllamaEmbedding sends a request to Ollama API to get embeddings for the given prompt
// Returns the embedding vector as []float32 and any error that occurred
func OllamaEmbedding(prompt, model string) ([]float32, error) {
	if prompt == "" || model == "" {
		return []float32{}, fmt.Errorf("empty prompt or model is not allowed")
	}

	// Prepare the request body
	reqBody := map[string]string{
		"model": model,
		"input": prompt,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request body: %v", err)
	}

	// Create the HTTP request
	req, err := http.NewRequest("POST", "http://localhost:11434/api/embed", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %v", err)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	// Parse the response
	var ollamaResp OllamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fmt.Errorf("error parsing response: %v", err)
	}

	// Check if we got any embeddings
	if len(ollamaResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned from Ollama")
	}

	return ollamaResp.Embeddings[0], nil
}
