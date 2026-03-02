package main

import (
	"log"
	"net/http"

	"governance-intel-platform/apps/api/internal/config"
	"governance-intel-platform/apps/api/internal/server"
)

func main() {
	cfg := config.Load()
	handler := server.NewRouter(cfg)

	log.Printf("apps/api listening on %s", cfg.Address)
	if err := http.ListenAndServe(cfg.Address, handler); err != nil {
		log.Fatal(err)
	}
}

