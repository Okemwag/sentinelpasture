package server

import (
	"encoding/json"
	"net/http"
	"time"

	"governance-intel-platform/apps/api/internal/config"
	"governance-intel-platform/apps/api/internal/service"
)

func NewRouter(cfg config.Config) http.Handler {
	mux := http.NewServeMux()
	dashboard := service.NewDashboardService()
	ai := service.NewAIGateway(cfg.AIInferenceURL)

	mux.HandleFunc("/", jsonHandler(func(r *http.Request) (any, error) {
		return map[string]any{
			"status":                  "operational",
			"service":                 "National Risk Intelligence API",
			"version":                 "1.0.0",
			"ai_engine_status":        ai.EngineName(),
			"timestamp":               nowRFC3339(),
		}, nil
	}))

	mux.HandleFunc("/health", jsonHandler(func(r *http.Request) (any, error) {
		return map[string]any{
			"status":                     "healthy",
			"ai_engine":                  ai.EngineName(),
			"ai_inference_url_configured": ai.RemoteEnabled(),
			"timestamp":                  nowRFC3339(),
		}, nil
	}))

	mux.HandleFunc("/api/dashboard/stability", jsonHandler(func(r *http.Request) (any, error) {
		return dashboard.Stability(), nil
	}))
	mux.HandleFunc("/api/dashboard/stats", jsonHandler(func(r *http.Request) (any, error) {
		return dashboard.Stats(), nil
	}))
	mux.HandleFunc("/api/drivers/list", jsonHandler(func(r *http.Request) (any, error) {
		return dashboard.Drivers(), nil
	}))
	mux.HandleFunc("/api/alerts/list", jsonHandler(func(r *http.Request) (any, error) {
		return dashboard.Alerts(), nil
	}))
	mux.HandleFunc("/api/regional/data", jsonHandler(func(r *http.Request) (any, error) {
		return dashboard.RegionalData(), nil
	}))

	mux.HandleFunc("/api/process", jsonHandler(func(r *http.Request) (any, error) {
		if r.Method != http.MethodPost {
			return methodNotAllowed(), nil
		}
		var payload struct {
			Signals []map[string]any `json:"signals"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			return badRequest("Invalid signals data"), nil
		}
		return ai.ProcessSignals(payload.Signals), nil
	}))

	mux.HandleFunc("/api/predict", jsonHandler(func(r *http.Request) (any, error) {
		region := r.URL.Query().Get("region")
		if region == "" {
			region = "national"
		}
		timeframe := r.URL.Query().Get("timeframe")
		if timeframe == "" {
			timeframe = "7d"
		}
		return ai.PredictRisk(region, timeframe), nil
	}))

	mux.HandleFunc("/api/analyze-drivers", jsonHandler(func(r *http.Request) (any, error) {
		if r.Method != http.MethodPost {
			return methodNotAllowed(), nil
		}
		var payload struct {
			Data map[string]any `json:"data"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			return badRequest("Invalid driver data"), nil
		}
		return ai.AnalyzeDrivers(payload.Data), nil
	}))

	mux.HandleFunc("/api/recommend-interventions", jsonHandler(func(r *http.Request) (any, error) {
		if r.Method != http.MethodPost {
			return methodNotAllowed(), nil
		}
		var payload struct {
			Region      string         `json:"region"`
			RiskProfile map[string]any `json:"riskProfile"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			return badRequest("Invalid intervention request"), nil
		}
		return ai.RecommendInterventions(payload.Region, payload.RiskProfile), nil
	}))

	return withCORS(mux)
}

func jsonHandler(fn func(r *http.Request) (any, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		payload, err := fn(r)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"success": false,
				"error":   err.Error(),
			})
			return
		}
		if status, ok := payload.(statusPayload); ok {
			w.WriteHeader(status.code)
			_ = json.NewEncoder(w).Encode(status.payload)
			return
		}
		_ = json.NewEncoder(w).Encode(payload)
	}
}

type statusPayload struct {
	code    int
	payload any
}

func badRequest(message string) statusPayload {
	return statusPayload{
		code: http.StatusBadRequest,
		payload: map[string]any{
			"success": false,
			"error":   message,
		},
	}
}

func methodNotAllowed() statusPayload {
	return statusPayload{
		code: http.StatusMethodNotAllowed,
		payload: map[string]any{
			"success": false,
			"error":   "Method not allowed",
		},
	}
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func nowRFC3339() string {
	return time.Now().UTC().Format(time.RFC3339)
}
