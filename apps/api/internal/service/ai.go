package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type AIGateway struct {
	BaseURL string
	Client  *http.Client
}

func NewAIGateway(baseURL string) AIGateway {
	return AIGateway{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Client: &http.Client{
			Timeout: 3 * time.Second,
		},
	}
}

func (g AIGateway) EngineName() string {
	if g.BaseURL != "" {
		return "contract_ai_service"
	}
	return "local_go_fallback_ai"
}

func (g AIGateway) RemoteEnabled() bool {
	return g.BaseURL != ""
}

func (g AIGateway) ProcessSignals(signals []map[string]any) map[string]any {
	if g.BaseURL == "" {
		return fallbackProcessSignals(signals)
	}

	regionID := regionFromSignals(signals)
	risk, err := g.postJSON("/infer/risk", map[string]any{
		"region_id": regionID,
		"at_time":   time.Now().UTC().Format(time.RFC3339),
		"signals":   signals,
	})
	if err != nil {
		return fallbackProcessSignals(signals)
	}

	explain, err := g.postJSON("/infer/explain", map[string]any{
		"region_id":   regionID,
		"risk_score":  number(risk["risk_score"]),
	})
	if err != nil {
		return fallbackProcessSignals(signals)
	}

	interventions, err := g.postJSON("/infer/interventions", map[string]any{
		"region_id":  regionID,
		"risk_score": number(risk["risk_score"]),
	})
	if err != nil {
		return fallbackProcessSignals(signals)
	}

	drivers := sliceOfMaps(risk["top_drivers"])
	items := sliceOfMaps(interventions["interventions"])

	indicators := make([]map[string]any, 0, len(drivers))
	for _, driver := range drivers {
		name := stringValue(driver["name"])
		indicators = append(indicators, map[string]any{
			"type":        strings.ToLower(strings.ReplaceAll(name, " ", "_")),
			"severity":    round(number(driver["contribution"])),
			"description": fmt.Sprintf("%s driving risk in %s; direction is %s", name, regionID, stringValue(driver["direction"])),
		})
	}

	recommendations := make([]map[string]any, 0, len(items))
	for index, item := range items {
		recommendations = append(recommendations, map[string]any{
			"action":           stringValue(item["category"]),
			"priority":         index + 1,
			"estimated_impact": stringValue(item["expected_impact"]),
		})
	}

	return map[string]any{
		"assessment": map[string]any{
			"threat_level": int(round(number(risk["risk_score"]) * 100)),
			"confidence":   number(risk["confidence"]),
			"risk_factors": driverNames(drivers),
		},
		"indicators":      indicators,
		"recommendations": recommendations,
		"metadata": map[string]any{
			"model_version":             stringValue(risk["model_version"]),
			"processing_time":           0.12,
			"engine":                    "contract_ai_service",
			"feature_snapshot_timestamp": risk["feature_snapshot_timestamp"],
			"known_data_gaps":           risk["known_data_gaps"],
			"explanation_summary":       stringValue(explain["summary"]),
		},
	}
}

func (g AIGateway) PredictRisk(region, timeframe string) map[string]any {
	if g.BaseURL == "" {
		return fallbackPredict(region, timeframe)
	}

	risk, err := g.postJSON("/infer/risk", map[string]any{
		"region_id": region,
		"at_time":   time.Now().UTC().Format(time.RFC3339),
		"signals":   []map[string]any{},
	})
	if err != nil {
		return fallbackPredict(region, timeframe)
	}

	days := parseDays(timeframe)
	base := int(round(number(risk["risk_score"]) * 100))
	predictions := make([]map[string]any, 0, min(days, 7))
	for offset := 0; offset < min(days, 7); offset++ {
		predictions = append(predictions, map[string]any{
			"date":       time.Now().UTC().AddDate(0, 0, offset).Format("2006-01-02"),
			"risk_level": min(base+offset, 99),
			"confidence": maxFloat(0.5, round(number(risk["confidence"])-(float64(offset)*0.02))),
		})
	}

	return map[string]any{
		"region":      region,
		"timeframe":   timeframe,
		"predictions": predictions,
		"confidence":  number(risk["confidence"]),
		"factors":     driverNames(sliceOfMaps(risk["top_drivers"])),
		"metadata": map[string]any{
			"model_version": stringValue(risk["model_version"]),
			"engine":        "contract_ai_service",
		},
	}
}

func (g AIGateway) AnalyzeDrivers(payload map[string]any) map[string]any {
	if g.BaseURL == "" {
		return fallbackAnalyzeDrivers(payload)
	}

	explain, err := g.postJSON("/infer/explain", map[string]any{
		"region_id":  stringOrDefault(payload["region_id"], "national"),
		"risk_score": numberOrDefault(payload["risk_score"], 0.6),
	})
	if err != nil {
		return fallbackAnalyzeDrivers(payload)
	}

	driversRaw := sliceOfMaps(explain["top_drivers"])
	drivers := make([]map[string]any, 0, len(driversRaw))
	for _, item := range driversRaw {
		trend := "stable"
		if stringValue(item["direction"]) == "up" {
			trend = "increasing"
		}
		drivers = append(drivers, map[string]any{
			"name":         stringValue(item["name"]),
			"contribution": round(number(item["contribution"])),
			"trend":        trend,
			"confidence":   0.84,
		})
	}

	relationships := []map[string]any{}
	if len(drivers) >= 2 {
		relationships = append(relationships, map[string]any{
			"from":     drivers[0]["name"],
			"to":       drivers[1]["name"],
			"strength": 0.61,
		})
	}

	return map[string]any{
		"drivers":               drivers,
		"causal_relationships":  relationships,
		"metadata": map[string]any{
			"model_version":    stringValue(explain["model_version"]),
			"engine":           "contract_ai_service",
			"uncertainty_notes": explain["uncertainty_notes"],
		},
	}
}

func (g AIGateway) RecommendInterventions(region string, riskProfile map[string]any) map[string]any {
	if g.BaseURL == "" {
		return fallbackRecommend(region, riskProfile)
	}

	riskScore := numberOrDefault(riskProfile["risk_score"], numberOrDefault(riskProfile["threat_level"], 55)/100)
	response, err := g.postJSON("/infer/interventions", map[string]any{
		"region_id":  region,
		"risk_score": riskScore,
	})
	if err != nil {
		return fallbackRecommend(region, riskProfile)
	}

	items := sliceOfMaps(response["interventions"])
	interventions := make([]map[string]any, 0, len(items))
	for _, item := range items {
		interventions = append(interventions, map[string]any{
			"category":            stringValue(item["category"]),
			"expectedImpact":      stringValue(item["expected_impact"]),
			"timeToEffect":        stringValue(item["time_to_effect"]),
			"costBand":            "Policy-defined",
			"confidence":          stringValue(item["confidence"]),
			"effectiveness_score": 0.78,
			"constraintsApplied":  item["constraints_applied"],
		})
	}

	return map[string]any{
		"region":        region,
		"interventions": interventions,
		"metadata": map[string]any{
			"model_version": stringValue(response["model_version"]),
			"engine":        "contract_ai_service",
		},
	}
}

func (g AIGateway) postJSON(path string, payload map[string]any) (map[string]any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, g.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := g.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("upstream status %s", resp.Status)
	}

	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func fallbackProcessSignals(signals []map[string]any) map[string]any {
	indicators := make([]map[string]any, 0, len(signals))
	for _, signal := range signals {
		signalType := stringOrDefault(signal["type"], "general")
		severity := scoreSignal(signalType, signal["data"])
		dataMap, _ := signal["data"].(map[string]any)
		keySummary := "limited telemetry"
		if len(dataMap) > 0 {
			keys := make([]string, 0, 2)
			for key := range dataMap {
				keys = append(keys, key)
				if len(keys) == 2 {
					break
				}
			}
			keySummary = strings.Join(keys, ", ")
		}
		indicators = append(indicators, map[string]any{
			"type":        signalType,
			"severity":    severity,
			"description": fmt.Sprintf("%s signal flagged from %s; severity scored at %.2f", strings.Title(strings.ReplaceAll(signalType, "_", " ")), keySummary, severity),
		})
	}

	threatLevel := 42
	if len(indicators) > 0 {
		total := 0.0
		for _, indicator := range indicators {
			total += number(indicator["severity"])
		}
		threatLevel = int(round((total / float64(len(indicators))) * 100))
	}

	riskFactors := make([]string, 0, min(len(indicators), 3))
	for index, indicator := range indicators {
		if index == 3 {
			break
		}
		riskFactors = append(riskFactors, strings.Title(strings.ReplaceAll(stringValue(indicator["type"]), "_", " ")))
	}
	if len(riskFactors) == 0 {
		riskFactors = []string{"Baseline monitoring"}
	}

	recommendations := []map[string]any{
		{"action": "Increase analyst review cadence", "priority": 1, "estimated_impact": impactBand(threatLevel)},
	}
	for _, indicator := range indicators {
		switch stringValue(indicator["type"]) {
		case "economic":
			recommendations = append(recommendations, map[string]any{"action": "Escalate economic stress monitoring", "priority": 2, "estimated_impact": "High"})
		case "environmental":
			recommendations = append(recommendations, map[string]any{"action": "Prepare resilience response coordination", "priority": 3, "estimated_impact": "Moderate"})
		}
	}

	return map[string]any{
		"assessment": map[string]any{
			"threat_level": threatLevel,
			"confidence":   round(minFloat(0.96, 0.58+(float64(len(indicators))*0.07))),
			"risk_factors": riskFactors,
		},
		"indicators":      indicators,
		"recommendations": recommendations,
		"metadata": map[string]any{
			"model_version": "local-dev-1.0",
			"processing_time": round(0.04 + (float64(len(signals)) * 0.015)),
			"engine": "local_go_fallback_ai",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		},
	}
}

func fallbackPredict(region, timeframe string) map[string]any {
	days := parseDays(timeframe)
	baseRisk := 48 + min(days, 14)
	predictions := make([]map[string]any, 0, min(days, 7))
	for offset := 0; offset < min(days, 7); offset++ {
		predictions = append(predictions, map[string]any{
			"date":       time.Now().UTC().AddDate(0, 0, offset).Format("2006-01-02"),
			"risk_level": min(95, baseRisk+offset),
			"confidence": maxFloat(0.62, round(0.85-(float64(offset)*0.02))),
		})
	}

	return map[string]any{
		"region":      region,
		"timeframe":   timeframe,
		"predictions": predictions,
		"confidence":  0.81,
		"factors":     []string{"Signal velocity", "Recent anomaly density", "Historical trend carry-over"},
		"metadata": map[string]any{
			"model_version": "local-dev-1.0",
			"engine":        "local_go_fallback_ai",
		},
	}
}

func fallbackAnalyzeDrivers(payload map[string]any) map[string]any {
	drivers := []map[string]any{
		{"name": "Baseline Stress", "contribution": 1.0, "trend": "increasing", "confidence": 0.84},
	}
	bestKey := ""
	bestValue := 0.0
	secondKey := ""
	secondValue := 0.0
	for key, raw := range payload {
		value := number(raw)
		if value <= 0 {
			continue
		}
		if value > bestValue {
			secondKey, secondValue = bestKey, bestValue
			bestKey, bestValue = key, value
		} else if value > secondValue {
			secondKey, secondValue = key, value
		}
	}

	if bestKey != "" {
		total := bestValue + secondValue
		if total == 0 {
			total = bestValue
		}
		drivers = []map[string]any{
			{"name": strings.Title(strings.ReplaceAll(bestKey, "_", " ")), "contribution": round(bestValue / total), "trend": "increasing", "confidence": 0.84},
		}
		if secondKey != "" {
			drivers = append(drivers, map[string]any{"name": strings.Title(strings.ReplaceAll(secondKey, "_", " ")), "contribution": round(secondValue / total), "trend": "stable", "confidence": 0.84})
		}
	}

	relationships := []map[string]any{}
	if len(drivers) >= 2 {
		relationships = append(relationships, map[string]any{
			"from":     drivers[0]["name"],
			"to":       drivers[1]["name"],
			"strength": 0.63,
		})
	}

	return map[string]any{
		"drivers":              drivers,
		"causal_relationships": relationships,
		"metadata": map[string]any{
			"model_version": "local-dev-1.0",
			"engine":        "local_go_fallback_ai",
		},
	}
}

func fallbackRecommend(region string, riskProfile map[string]any) map[string]any {
	threatLevel := int(numberOrDefault(riskProfile["threat_level"], 55))
	expectedImpact := "Moderate"
	if threatLevel >= 70 {
		expectedImpact = "High"
	}

	return map[string]any{
		"region": region,
		"interventions": []map[string]any{
			{
				"category":            "Targeted field assessment",
				"expectedImpact":      expectedImpact,
				"timeToEffect":        "Short",
				"costBand":            "Low",
				"confidence":          "High",
				"effectiveness_score": 0.83,
			},
			{
				"category":            "Multi-agency coordination cell",
				"expectedImpact":      "High",
				"timeToEffect":        "Medium",
				"costBand":            "Medium",
				"confidence":          "High",
				"effectiveness_score": 0.79,
			},
		},
		"metadata": map[string]any{
			"model_version": "local-dev-1.0",
			"engine":        "local_go_fallback_ai",
		},
	}
}

func regionFromSignals(signals []map[string]any) string {
	for _, signal := range signals {
		location, ok := signal["location"].(map[string]any)
		if !ok {
			continue
		}
		if region := stringValue(location["region"]); region != "" {
			return region
		}
		if name := stringValue(location["name"]); name != "" {
			return name
		}
	}
	return "national"
}

func driverNames(drivers []map[string]any) []string {
	names := make([]string, 0, len(drivers))
	for _, driver := range drivers {
		names = append(names, stringValue(driver["name"]))
	}
	return names
}

func parseDays(timeframe string) int {
	cleaned := strings.TrimSpace(strings.ToLower(timeframe))
	if strings.HasSuffix(cleaned, "d") {
		if value, err := strconv.Atoi(strings.TrimSuffix(cleaned, "d")); err == nil && value > 0 {
			return value
		}
	}
	return 7
}

func scoreSignal(signalType string, data any) float64 {
	score := 0.35
	dataMap, _ := data.(map[string]any)
	numericCount := 0
	numericTotal := 0.0
	textParts := make([]string, 0, len(dataMap))
	for _, raw := range dataMap {
		switch value := raw.(type) {
		case float64:
			numericTotal += value
			numericCount++
		case int:
			numericTotal += float64(value)
			numericCount++
		case string:
			textParts = append(textParts, strings.ToLower(value))
		}
	}
	if numericCount > 0 {
		score += minFloat((numericTotal/float64(numericCount))/200.0, 0.35)
	}
	textBlob := strings.Join(textParts, " ")
	for _, keyword := range []struct {
		word   string
		weight float64
	}{
		{"protest", 0.12},
		{"shortage", 0.10},
		{"violence", 0.18},
		{"drought", 0.09},
		{"price", 0.07},
		{"unemployment", 0.08},
	} {
		if strings.Contains(textBlob, keyword.word) {
			score += keyword.weight
		}
	}
	switch signalType {
	case "economic":
		score += 0.08
	case "environmental":
		score += 0.06
	case "social":
		score += 0.07
	case "security":
		score += 0.12
	default:
		score += 0.03
	}
	return round(minFloat(score, 0.98))
}

func impactBand(threatLevel int) string {
	if threatLevel >= 65 {
		return "High"
	}
	return "Moderate"
}

func sliceOfMaps(raw any) []map[string]any {
	items, ok := raw.([]any)
	if !ok {
		if typed, ok := raw.([]map[string]any); ok {
			return typed
		}
		return []map[string]any{}
	}
	result := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if mapped, ok := item.(map[string]any); ok {
			result = append(result, mapped)
		}
	}
	return result
}

func stringValue(raw any) string {
	if value, ok := raw.(string); ok {
		return value
	}
	return ""
}

func stringOrDefault(raw any, fallback string) string {
	value := stringValue(raw)
	if value == "" {
		return fallback
	}
	return value
}

func number(raw any) float64 {
	switch value := raw.(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		parsed, _ := value.Float64()
		return parsed
	default:
		return 0
	}
}

func numberOrDefault(raw any, fallback float64) float64 {
	value := number(raw)
	if value == 0 {
		return fallback
	}
	return value
}

func round(value float64) float64 {
	return float64(int(value*100+0.5)) / 100
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
