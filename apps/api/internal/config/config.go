package config

import "os"

type Config struct {
	Address       string
	AIInferenceURL string
}

func Load() Config {
	address := os.Getenv("API_ADDR")
	if address == "" {
		address = ":8000"
	}

	aiInferenceURL := os.Getenv("AI_INFERENCE_URL")

	return Config{
		Address:       address,
		AIInferenceURL: aiInferenceURL,
	}
}
