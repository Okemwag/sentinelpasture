package health

import (
	"encoding/json"
	"net/http"
	"time"
)

type Response struct {
	Service   string `json:"service"`
	Status    string `json:"status"`
	Timestamp string `json:"timestamp"`
}

func Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Response{
			Service:   "governance-intel-api",
			Status:    "ok",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		})
	}
}

