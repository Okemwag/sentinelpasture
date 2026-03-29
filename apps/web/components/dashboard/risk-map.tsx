"use client";

import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { apiClient } from "@/lib/api-client";
import {
  KENYA_OUTLINE_PATH,
  KENYA_REGION_SHAPES,
  getRiskPalette,
  type RegionalRiskLevel,
} from "@/lib/kenya-map";

export type MapRegion = {
  id: string;
  name: string;
  shapeId: string;
  riskLevel: RegionalRiskLevel;
  riskScore: number;
  thresholdStatus: string;
  primaryDriver: string;
  secondaryDriver: string;
  confidence: string;
  storySummary: string;
  thresholdReason: string;
  communityContext: string;
  watchItems: string[];
  sourceRegionId: string;
  population?: string;
};

export type MapEventPoint = {
  id: number;
  title: string;
  description: string;
  timestamp: string;
  region: string;
  severity: "elevated" | "moderate";
  status: "active" | "monitoring" | "resolved";
  sourceRegionId?: string;
};

interface RiskMapProps {
  onRegionSelect?: (region: MapRegion) => void;
  selectedRegionId?: string | null;
  activeLayer?: "risk" | "events" | "mobility" | "satellite";
  timeWindow?: "7d" | "30d" | "90d";
  satelliteTimestamp?: string;
  satelliteCompare?: number;
  alerts?: Array<{
    id: number;
    title: string;
    description: string;
    timestamp: string;
    region: string;
    severity: "elevated" | "moderate";
    status: "active" | "monitoring" | "resolved";
  }>;
  playbackProgress?: number;
  onEventSelect?: (eventPoint: MapEventPoint) => void;
  satelliteTileTemplate?: string;
}

export default function RiskMap({
  onRegionSelect,
  selectedRegionId,
  activeLayer = "risk",
  timeWindow = "30d",
  satelliteTimestamp,
  satelliteCompare = 50,
  alerts = [],
  playbackProgress = 100,
  onEventSelect,
  satelliteTileTemplate,
}: RiskMapProps) {
  const [regions, setRegions] = useState<MapRegion[]>([]);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const hasAutoSelected = useRef(false);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await apiClient.getRegionalMap();
        if (!active) return;
        const sorted = [...response.data].sort((a, b) => b.riskScore - a.riskScore);
        setRegions(sorted);
        if (!hasAutoSelected.current && sorted[0]) {
          hasAutoSelected.current = true;
          onRegionSelect?.(sorted[0]);
        }
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load regional map");
        // Load placeholders
        setRegions(PLACEHOLDER_REGIONS);
        if (!hasAutoSelected.current && PLACEHOLDER_REGIONS[0]) {
          hasAutoSelected.current = true;
          onRegionSelect?.(PLACEHOLDER_REGIONS[0]);
        }
      }
    }
    void load();
    return () => { active = false; };
  }, []);

  const visibleRegions = regions.length ? regions : PLACEHOLDER_REGIONS;
  const activeId = selectedRegionId ?? visibleRegions[0]?.id;

  // Find highest risk region for pulse ring
  const criticalRegion = visibleRegions.find((r) => r.riskLevel === "critical")
    ?? visibleRegions.find((r) => r.riskLevel === "elevated")
    ?? visibleRegions[0];
  const hasMapboxToken = Boolean(process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN);

  if (hasMapboxToken) {
    return (
      <MapboxRiskMap
        regions={visibleRegions}
        error={error}
        selectedRegionId={selectedRegionId}
        onRegionSelect={onRegionSelect}
        activeLayer={activeLayer}
        timeWindow={timeWindow}
        satelliteTimestamp={satelliteTimestamp}
        satelliteCompare={satelliteCompare}
        alerts={alerts}
        playbackProgress={playbackProgress}
        onEventSelect={onEventSelect}
        satelliteTileTemplate={satelliteTileTemplate}
      />
    );
  }

  return (
    <div className="relative h-full w-full flex flex-col">
      {error && (
        <div className="mb-2 text-[12px] text-[#8C6A3D] bg-[#FBF5EB] border border-[#E2C99A] rounded-[6px] px-3 py-2">
          Live data unavailable — showing modelled baseline
        </div>
      )}

      <LayerLegend activeLayer={activeLayer} satelliteTimestamp={satelliteTimestamp} />

      {/* Map SVG */}
      <div className="flex-1 relative">
        <svg
          viewBox="70 20 280 520"
          className="h-full w-full"
          style={{ minHeight: 400 }}
        >
          {/* Country outline shadow */}
          <path
            d={KENYA_OUTLINE_PATH}
            fill="none"
            stroke="#C8D8C2"
            strokeWidth="14"
            strokeLinejoin="round"
            strokeOpacity={0.4}
          />
          {/* Country outline */}
          <path
            d={KENYA_OUTLINE_PATH}
            fill="#F7F7F5"
            stroke="#9CA3AF"
            strokeWidth="3"
            strokeLinejoin="round"
          />

          {visibleRegions.map((region) => {
            const shape = KENYA_REGION_SHAPES[region.shapeId];
            if (!shape) return null;
            const palette = getRiskPalette(region.riskLevel);
            const isActive = activeId === region.id;
            const isHovered = hoveredId === region.id;
            const isPulse = criticalRegion?.id === region.id;

            // Compute centroid from polygon points for the pulse dot
            const pts = shape.points.split(" ").map((p) => {
              const [x, y] = p.split(",").map(Number);
              return { x, y };
            });
            const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
            const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;

            return (
              <g
                key={region.id}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredId(region.id)}
                onMouseLeave={() => setHoveredId(null)}
                onClick={() => onRegionSelect?.(region)}
              >
                <polygon
                  points={shape.points}
                  fill={activeLayer === "events" ? "#FCA5A5" : activeLayer === "mobility" ? "#99F6E4" : palette.fill}
                  stroke={isActive ? "#111111" : isHovered ? "#374151" : palette.stroke}
                  strokeWidth={isActive ? 3 : isHovered ? 2 : 1.5}
                  opacity={isActive ? 1 : isHovered ? 0.95 : 0.85}
                  style={{ transition: "all 0.2s ease" }}
                />

                {/* Region abbreviation label */}
                <text
                  x={shape.labelX}
                  y={shape.labelY}
                  textAnchor="middle"
                  fontSize={isActive ? 10 : 9}
                  fontWeight={isActive ? 700 : 500}
                  fill={isActive ? "#111111" : "#374151"}
                  style={{ pointerEvents: "none", letterSpacing: "0.08em" }}
                >
                  {shape.label}
                </text>

                {/* Pulse ring for the highest-risk region */}
                {isPulse && (
                  <>
                    <circle
                      cx={cx}
                      cy={cy}
                      r={10}
                      fill={palette.fill}
                      fillOpacity={0.3}
                      style={{
                        transformOrigin: `${cx}px ${cy}px`,
                        animation: "pulseRing 2s ease-out infinite",
                        pointerEvents: "none",
                      }}
                    />
                    <circle cx={cx} cy={cy} r={4} fill={palette.stroke} />
                  </>
                )}
                {activeLayer === "events" && (
                  <circle cx={cx} cy={cy} r={Math.max(3, region.riskScore * 8)} fill="#B91C1C" fillOpacity={0.35} />
                )}
                {activeLayer === "mobility" && (
                  <circle cx={cx} cy={cy} r={Math.max(3, region.riskScore * 7)} fill="#0F766E" fillOpacity={0.32} />
                )}
              </g>
            );
          })}
        </svg>

        {/* Hover tooltip */}
        {hoveredId && (() => {
          const r = visibleRegions.find((x) => x.id === hoveredId);
          if (!r) return null;
          const pal = getRiskPalette(r.riskLevel);
          return (
            <div
              className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-white border border-[#E5E7EB] rounded-[8px] px-3 py-2 shadow-lg text-[12px] pointer-events-none z-10 whitespace-nowrap"
              style={{ boxShadow: "0 4px 16px rgba(0,0,0,0.10)" }}
            >
              <span className="font-semibold text-[#111111]">{r.name}</span>
              <span
                className="ml-2 px-1.5 py-0.5 rounded-full text-[10px] font-semibold border"
                style={{ backgroundColor: pal.fill, color: pal.textColor, borderColor: pal.stroke }}
              >
                {pal.label}
              </span>
              <span className="ml-2 text-[#6B7280]">
                Score: {Math.round(r.riskScore * 100)}
              </span>
            </div>
          );
        })()}
      </div>

      {/* Bottom risk-level bar */}
      <div className="mt-3 flex gap-1 h-1.5 rounded-full overflow-hidden">
        {visibleRegions.map((r) => {
          const pal = getRiskPalette(r.riskLevel);
          return (
            <div
              key={r.id}
              className="flex-1 cursor-pointer transition-opacity"
              style={{ backgroundColor: pal.fill, opacity: activeId === r.id ? 1 : 0.5 }}
              onClick={() => onRegionSelect?.(r)}
              title={r.name}
            />
          );
        })}
      </div>
    </div>
  );
}

function MapboxRiskMap({
  regions,
  error,
  selectedRegionId,
  onRegionSelect,
  activeLayer,
  timeWindow,
  satelliteTimestamp,
  satelliteCompare,
  alerts,
  playbackProgress,
  onEventSelect,
  satelliteTileTemplate,
}: {
  regions: MapRegion[];
  error: string;
  selectedRegionId?: string | null;
  onRegionSelect?: (region: MapRegion) => void;
  activeLayer: "risk" | "events" | "mobility" | "satellite";
  timeWindow: "7d" | "30d" | "90d";
  satelliteTimestamp?: string;
  satelliteCompare?: number;
  alerts: Array<{
    id: number;
    title: string;
    description: string;
    timestamp: string;
    region: string;
    severity: "elevated" | "moderate";
    status: "active" | "monitoring" | "resolved";
  }>;
  playbackProgress: number;
  onEventSelect?: (eventPoint: MapEventPoint) => void;
  satelliteTileTemplate?: string;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const markersRef = useRef<mapboxgl.Marker[]>([]);
  const token = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN;

  useEffect(() => {
    if (!mapContainerRef.current || !token || mapRef.current) return;
    mapboxgl.accessToken = token;
    mapRef.current = new mapboxgl.Map({
      container: mapContainerRef.current,
      style:
        activeLayer === "satellite"
          ? "mapbox://styles/mapbox/satellite-streets-v12"
          : "mapbox://styles/mapbox/light-v11",
      center: [37.9, 0.5],
      zoom: 5.2,
      attributionControl: false,
    });
    mapRef.current.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "bottom-right");
    mapRef.current.on("load", () => {
      upsertSatelliteRasterLayer(mapRef.current, satelliteTileTemplate);
      upsertOperationalLayers(mapRef.current, regions, alerts, timeWindow, playbackProgress);
      applyLayerVisibility(mapRef.current, activeLayer);
    });
    mapRef.current.on("styledata", () => {
      upsertSatelliteRasterLayer(mapRef.current, satelliteTileTemplate);
      upsertOperationalLayers(mapRef.current, regions, alerts, timeWindow, playbackProgress);
      applyLayerVisibility(mapRef.current, activeLayer);
    });
    return () => {
      markersRef.current.forEach((m) => m.remove());
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [token, activeLayer, regions, alerts, timeWindow, playbackProgress, satelliteTileTemplate]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.setStyle(
      activeLayer === "satellite"
        ? "mapbox://styles/mapbox/satellite-streets-v12"
        : "mapbox://styles/mapbox/light-v11"
    );
    const onStyleReady = () => {
      upsertSatelliteRasterLayer(map, satelliteTileTemplate);
      upsertOperationalLayers(map, regions, alerts, timeWindow, playbackProgress);
      applyLayerVisibility(map, activeLayer);
    };
    if (map.isStyleLoaded()) {
      onStyleReady();
    } else {
      map.once("idle", onStyleReady);
    }
    return () => {
      map.off("idle", onStyleReady);
    };
  }, [activeLayer, regions, alerts, timeWindow, playbackProgress, satelliteTileTemplate]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!map.isStyleLoaded()) return;
    upsertSatelliteRasterLayer(map, satelliteTileTemplate);
    upsertOperationalLayers(map, regions, alerts, timeWindow, playbackProgress);
    applyLayerVisibility(map, activeLayer);
  }, [regions, alerts, timeWindow, playbackProgress, activeLayer, satelliteTileTemplate]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const clickHandler = (event: mapboxgl.MapLayerMouseEvent) => {
      const feature = event.features?.[0];
      if (!feature) return;
      const props = feature.properties as Record<string, string | number | undefined>;
      const name = String(props.region ?? "Unknown region");
      const title = String(props.title ?? "Incident");
      const severity = String(props.severity ?? "monitoring");
      const timestamp = String(props.timestamp ?? "");
      const description = String(props.description ?? "");
      const id = Number(props.id ?? 0);
      const status = String(props.status ?? "monitoring") as MapEventPoint["status"];
      const coordinates = (feature.geometry as GeoJSON.Point).coordinates;
      onEventSelect?.({
        id,
        title,
        description,
        timestamp,
        region: name,
        severity: severity === "elevated" ? "elevated" : "moderate",
        status,
        sourceRegionId: typeof props.sourceRegionId === "string" ? props.sourceRegionId : undefined,
      });
      new mapboxgl.Popup({ closeButton: true, closeOnClick: true })
        .setLngLat([coordinates[0], coordinates[1]])
        .setHTML(
          `<div style="font-family: Inter, sans-serif; min-width: 220px;">
            <div style="font-size:11px;color:#6B7280;margin-bottom:4px;">${name}</div>
            <div style="font-size:13px;font-weight:600;color:#111827;">${title}</div>
            <div style="font-size:11px;color:#B91C1C;margin:4px 0;">${severity.toUpperCase()}</div>
            <div style="font-size:11px;color:#6B7280;">${timestamp}</div>
            <div style="font-size:12px;color:#374151;margin-top:6px;">${description}</div>
          </div>`
        )
        .addTo(map);
    };
    if (map.getLayer("event-points-layer")) {
      map.on("click", "event-points-layer", clickHandler);
      map.on("mouseenter", "event-points-layer", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "event-points-layer", () => {
        map.getCanvas().style.cursor = "";
      });
    }
    return () => {
      // Style/map may already be torn down during hot reload or style swap.
      try {
        map.off("click", "event-points-layer", clickHandler);
      } catch {
        // noop
      }
    };
  }, [activeLayer, alerts, onEventSelect, satelliteTileTemplate, regions, timeWindow, playbackProgress]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    regions.forEach((region) => {
      const coords = REGION_COORDS[region.id];
      if (!coords) return;
      const palette = getRiskPalette(region.riskLevel);
      const el = document.createElement("button");
      el.type = "button";
      el.style.width = selectedRegionId === region.id ? "18px" : "14px";
      el.style.height = selectedRegionId === region.id ? "18px" : "14px";
      el.style.borderRadius = "9999px";
      const markerColor =
        activeLayer === "events"
          ? "#B91C1C"
          : activeLayer === "mobility"
            ? "#0F766E"
            : palette.fill;
      el.style.border = selectedRegionId === region.id ? "2px solid #111111" : `2px solid ${palette.stroke}`;
      el.style.background = markerColor;
      el.style.cursor = "pointer";
      el.style.boxShadow = "0 2px 8px rgba(15,23,42,0.2)";
      el.title = `${region.name} · ${Math.round(region.riskScore * 100)}`;
      el.onclick = () => onRegionSelect?.(region);
      const marker = new mapboxgl.Marker({ element: el }).setLngLat(coords).addTo(map);
      markersRef.current.push(marker);
    });
  }, [regions, selectedRegionId, onRegionSelect]);

  return (
    <div className="relative h-full w-full flex flex-col">
      {error && (
        <div className="mb-2 text-[12px] text-[#8C6A3D] bg-[#FBF5EB] border border-[#E2C99A] rounded-[6px] px-3 py-2">
          Live data unavailable — showing modelled baseline
        </div>
      )}
      <LayerLegend activeLayer={activeLayer} satelliteTimestamp={satelliteTimestamp} />
      <div className="mb-2 text-[11px] text-[#6B7280]">
        Geospatial mode · {activeLayer} layer · {timeWindow} window
      </div>
      <div className="relative w-full flex-1">
        <div ref={mapContainerRef} className="w-full h-full rounded-[8px] border border-[#E5E7EB]" style={{ minHeight: 420 }} />
        {activeLayer === "satellite" && (
          <div
            className="absolute inset-y-0 left-0 pointer-events-none rounded-l-[8px] border-r-2 border-white/80"
            style={{
              width: `${Math.max(5, Math.min(95, satelliteCompare ?? 50))}%`,
              background: "linear-gradient(90deg, rgba(37,99,235,0.14), rgba(37,99,235,0.04))",
            }}
          />
        )}
      </div>
      <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
        {regions.slice(0, 4).map((r) => {
          const pal = getRiskPalette(r.riskLevel);
          return (
            <button
              key={r.id}
              type="button"
              onClick={() => onRegionSelect?.(r)}
              className="text-left rounded-[8px] border border-[#E5E7EB] px-2.5 py-2 hover:bg-[#FAFAFA]"
            >
              <div className="flex items-center justify-between">
                <span className="text-[12px] text-[#111111]">{r.name}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full border" style={{ borderColor: pal.stroke, color: pal.textColor }}>
                  {Math.round(r.riskScore * 100)}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LayerLegend({
  activeLayer,
  satelliteTimestamp,
}: {
  activeLayer: "risk" | "events" | "mobility" | "satellite";
  satelliteTimestamp?: string;
}) {
  if (activeLayer === "events") {
    return (
      <div className="flex flex-wrap gap-2 mb-3">
        <LegendChip color="#B91C1C" label="Event pressure intensity" />
        <LegendChip color="#FCA5A5" label="High event concentration" />
        <LegendChip color="#9CA3AF" label="Lower event concentration" />
      </div>
    );
  }
  if (activeLayer === "mobility") {
    return (
      <div className="flex flex-wrap gap-2 mb-3">
        <LegendChip color="#0F766E" label="Mobility pressure index" />
        <LegendChip color="#99F6E4" label="Corridor stress zones" />
      </div>
    );
  }
  if (activeLayer === "satellite") {
    return (
      <div className="flex flex-wrap gap-2 mb-3 items-center">
        <LegendChip color="#2563EB" label="Satellite basemap active" />
        <span className="text-[11px] text-[#374151]">
          Timestamp: {satelliteTimestamp ?? "latest available"}
        </span>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-2 mb-3">
      {(["low", "watch", "elevated", "critical"] as const).map((level, idx) => {
        const pal = getRiskPalette(level);
        const labels = ["Low", "Elevated", "High", "Critical"];
        return <LegendChip key={level} color={pal.fill} border={pal.stroke} label={labels[idx]} />;
      })}
    </div>
  );
}

function LegendChip({ color, label, border }: { color: string; label: string; border?: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 text-[11px] text-[#374151]">
      <span
        className="inline-block h-3 w-3 rounded-sm shrink-0"
        style={{ backgroundColor: color, border: `1px solid ${border ?? color}` }}
      />
      {label}
    </div>
  );
}

function upsertSatelliteRasterLayer(map: mapboxgl.Map | null, tileTemplate?: string) {
  if (!map || !map.isStyleLoaded() || !tileTemplate) return;
  if (map.getLayer("gibs-satellite-layer")) {
    map.removeLayer("gibs-satellite-layer");
  }
  if (map.getSource("gibs-satellite-source")) {
    map.removeSource("gibs-satellite-source");
  }
  map.addSource("gibs-satellite-source", {
    type: "raster",
    tiles: [tileTemplate],
    tileSize: 256,
  });
  map.addLayer(
    {
      id: "gibs-satellite-layer",
      type: "raster",
      source: "gibs-satellite-source",
      paint: {
        "raster-opacity": 0.85,
        "raster-saturation": -0.1,
        "raster-contrast": 0.04,
      },
    },
    "regional-alert-clusters"
  );
}

function upsertOperationalLayers(
  map: mapboxgl.Map | null,
  regions: MapRegion[],
  alerts: Array<{
    id: number;
    title: string;
    description: string;
    timestamp: string;
    region: string;
    severity: "elevated" | "moderate";
    status: "active" | "monitoring" | "resolved";
  }>,
  timeWindow: "7d" | "30d" | "90d",
  playbackProgress: number
) {
  if (!map) return;
  if (!map.isStyleLoaded()) return;
  const filteredAlerts = filterAlertsForWindow(alerts, timeWindow, playbackProgress);
  const points = regions
    .map((region) => {
      const coords = REGION_COORDS[region.id];
      if (!coords) return null;
      return {
        type: "Feature",
        properties: {
          id: region.id,
          name: region.name,
          weight: Math.round(region.riskScore * 100),
          risk: region.riskLevel,
        },
        geometry: {
          type: "Point",
          coordinates: coords,
        },
      };
    })
    .filter(Boolean) as GeoJSON.Feature<GeoJSON.Point>[];
  const sourceData: GeoJSON.FeatureCollection<GeoJSON.Point> = {
    type: "FeatureCollection",
    features: points,
  };
  if (!map.getSource("regional-points")) {
    map.addSource("regional-points", { type: "geojson", data: sourceData });
  } else {
    const source = map.getSource("regional-points") as mapboxgl.GeoJSONSource;
    source.setData(sourceData);
  }

  if (!map.getLayer("regional-events-heat")) {
    map.addLayer({
      id: "regional-events-heat",
      type: "heatmap",
      source: "regional-points",
      paint: {
        "heatmap-weight": ["interpolate", ["linear"], ["get", "weight"], 0, 0, 100, 1],
        "heatmap-intensity": 1.15,
        "heatmap-radius": 24,
        "heatmap-opacity": 0.8,
        "heatmap-color": [
          "interpolate",
          ["linear"],
          ["heatmap-density"],
          0,
          "rgba(252,165,165,0)",
          0.35,
          "rgba(248,113,113,0.35)",
          0.7,
          "rgba(220,38,38,0.55)",
          1,
          "rgba(153,27,27,0.78)",
        ],
      },
    });
  }

  const eventPoints = buildEventPoints(filteredAlerts, regions);
  if (!map.getSource("event-points")) {
    map.addSource("event-points", { type: "geojson", data: eventPoints });
  } else {
    const source = map.getSource("event-points") as mapboxgl.GeoJSONSource;
    source.setData(eventPoints);
  }
  if (!map.getLayer("event-points-layer")) {
    map.addLayer({
      id: "event-points-layer",
      type: "circle",
      source: "event-points",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["get", "weight"], 0, 4, 100, 12],
        "circle-color": [
          "case",
          ["==", ["get", "severity"], "elevated"],
          "#B91C1C",
          "#F59E0B",
        ],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 1.2,
        "circle-opacity": 0.85,
      },
    });
  }

  if (!map.getLayer("regional-alert-clusters")) {
    map.addLayer({
      id: "regional-alert-clusters",
      type: "circle",
      source: "regional-points",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["get", "weight"], 0, 5, 100, 16],
        "circle-color": [
          "interpolate",
          ["linear"],
          ["get", "weight"],
          0,
          "#A7F3D0",
          40,
          "#FDE68A",
          70,
          "#F59E0B",
          100,
          "#B91C1C",
        ],
        "circle-opacity": 0.65,
        "circle-stroke-color": "#111827",
        "circle-stroke-width": 1,
      },
    });
  }

  if (!map.getSource("mobility-links")) {
    map.addSource("mobility-links", { type: "geojson", data: buildMobilityLinks(regions) });
  } else {
    const source = map.getSource("mobility-links") as mapboxgl.GeoJSONSource;
    source.setData(buildMobilityLinks(regions));
  }

  if (!map.getLayer("regional-mobility-links")) {
    map.addLayer({
      id: "regional-mobility-links",
      type: "line",
      source: "mobility-links",
      paint: {
        "line-color": "#0F766E",
        "line-width": ["interpolate", ["linear"], ["get", "weight"], 0, 1.5, 100, 5],
        "line-opacity": 0.55,
      },
    });
  }
}

function applyLayerVisibility(map: mapboxgl.Map | null, layer: "risk" | "events" | "mobility" | "satellite") {
  if (!map) return;
  const set = (id: string, visible: boolean) => {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    }
  };
  set("regional-events-heat", layer === "events");
  set("regional-alert-clusters", layer === "events" || layer === "risk");
  set("event-points-layer", layer === "events");
  set("regional-mobility-links", layer === "mobility");
  set("gibs-satellite-layer", layer === "satellite");
}

function buildMobilityLinks(regions: MapRegion[]): GeoJSON.FeatureCollection<GeoJSON.LineString> {
  const hotspots = [...regions]
    .filter((r) => r.riskScore >= 0.4)
    .sort((a, b) => b.riskScore - a.riskScore)
    .slice(0, 4);
  const lines: GeoJSON.Feature<GeoJSON.LineString>[] = [];
  for (let i = 0; i < hotspots.length - 1; i++) {
    const from = REGION_COORDS[hotspots[i].id];
    const to = REGION_COORDS[hotspots[i + 1].id];
    if (!from || !to) continue;
    lines.push({
      type: "Feature",
      properties: {
        weight: Math.round(((hotspots[i].riskScore + hotspots[i + 1].riskScore) / 2) * 100),
      },
      geometry: {
        type: "LineString",
        coordinates: [from, to],
      },
    });
  }
  return { type: "FeatureCollection", features: lines };
}

function filterAlertsForWindow(
  alerts: Array<{
    id: number;
    title: string;
    description: string;
    timestamp: string;
    region: string;
    severity: "elevated" | "moderate";
    status: "active" | "monitoring" | "resolved";
  }>,
  window: "7d" | "30d" | "90d",
  progress: number
) {
  const days = window === "7d" ? 7 : window === "30d" ? 30 : 90;
  const now = Date.now();
  const start = now - days * 24 * 60 * 60 * 1000;
  const playbackTs = start + ((Math.max(0, Math.min(100, progress)) / 100) * (now - start));
  return alerts.filter((a) => {
    const ts = Number(new Date(a.timestamp));
    return ts >= start && ts <= playbackTs;
  });
}

function buildEventPoints(
  alerts: Array<{
    id: number;
    title: string;
    description: string;
    timestamp: string;
    region: string;
    severity: "elevated" | "moderate";
    status: "active" | "monitoring" | "resolved";
  }>,
  regions: MapRegion[]
): GeoJSON.FeatureCollection<GeoJSON.Point> {
  const regionByName = new Map(regions.map((r) => [r.name.toLowerCase(), r]));
  const features: GeoJSON.Feature<GeoJSON.Point>[] = [];
  alerts.forEach((alert, idx) => {
    const mappedRegion = regionByName.get(alert.region.toLowerCase());
    const base = mappedRegion ? REGION_COORDS[mappedRegion.id] : undefined;
    if (!base) return;
    const jitterLng = ((idx % 5) - 2) * 0.08;
    const jitterLat = ((idx % 7) - 3) * 0.06;
    features.push({
      type: "Feature",
      properties: {
        id: alert.id,
        title: alert.title,
        description: alert.description,
        region: alert.region,
        severity: alert.severity,
        status: alert.status,
        timestamp: new Date(alert.timestamp).toLocaleString(),
        weight: alert.severity === "elevated" ? 90 : 55,
        sourceRegionId: mappedRegion?.sourceRegionId ?? "",
      },
      geometry: {
        type: "Point",
        coordinates: [base[0] + jitterLng, base[1] + jitterLat],
      },
    });
  });
  return { type: "FeatureCollection", features };
}

const REGION_COORDS: Record<string, [number, number]> = {
  north_west_frontier: [35.5, 2.5],
  north_eastern_drylands: [40.4, 2.0],
  upper_eastern_corridor: [38.0, 0.3],
  lake_basin: [34.8, -0.1],
  central_highlands: [37.2, -0.4],
  nairobi_metro: [36.82, -1.29],
  coast_belt: [39.6, -3.4],
  south_rift: [35.8, -1.7],
};

const PLACEHOLDER_REGIONS: MapRegion[] = [
  {
    id: "north_eastern_drylands",
    shapeId: "north_eastern_drylands",
    name: "North Eastern Drylands",
    riskLevel: "elevated",
    riskScore: 0.74,
    thresholdStatus: "High threshold crossed",
    primaryDriver: "Rainfall anomaly +210%",
    secondaryDriver: "Pasture shocks, herd movement, borderland market timing",
    confidence: "Medium",
    storySummary:
      "Abrupt rainfall shifts are pulling herds toward a smaller number of viable water points before markets and local services can adjust.",
    thresholdReason:
      "Uneven rainfall changes grazing routes and local price pressure before it appears as a visible incident spike.",
    communityContext:
      "Many communities across this belt are pastoral and Cushitic-speaking, so rainfall shifts quickly affect livestock movement and support routes.",
    watchItems: [
      "Short-horizon rainfall deviation and pasture concentration",
      "Livestock price swings and distress sales",
      "Water-point pressure and local mediation requests",
    ],
    sourceRegionId: "placeholder",
    population: "841,000",
  },
  {
    id: "north_west_frontier",
    shapeId: "north_west_frontier",
    name: "North West Frontier",
    riskLevel: "watch",
    riskScore: 0.52,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Drought onset",
    secondaryDriver: "Cross-border livestock movement",
    confidence: "Low",
    storySummary: "Emerging drought conditions are beginning to affect pastoral routes.",
    thresholdReason: "Drought onset indicators rising above seasonal norms.",
    communityContext: "Predominantly agro-pastoral communities with seasonal migration patterns.",
    watchItems: ["Rainfall accumulation deficit", "Livestock body condition scores"],
    sourceRegionId: "placeholder",
    population: "330,000",
  },
  {
    id: "lake_basin",
    shapeId: "lake_basin",
    name: "Lake Basin Region",
    riskLevel: "watch",
    riskScore: 0.47,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Flooding risk",
    secondaryDriver: "Resource competition",
    confidence: "Medium",
    storySummary: "Elevated lake levels are creating localized flooding risk.",
    thresholdReason: "Lake levels approaching seasonal flood thresholds.",
    communityContext: "Fishing and farming communities at lake margins.",
    watchItems: ["Lake water levels", "Fish market pricing"],
    sourceRegionId: "placeholder",
    population: "980,000",
  },
  {
    id: "upper_eastern_corridor",
    shapeId: "upper_eastern_corridor",
    name: "Upper Eastern Corridor",
    riskLevel: "low",
    riskScore: 0.28,
    thresholdStatus: "Below threshold",
    primaryDriver: "Seasonal drought risk",
    secondaryDriver: "Land conflict",
    confidence: "High",
    storySummary: "Currently stable with low-level monitoring.",
    thresholdReason: "Seasonal rainfall expectations met.",
    communityContext: "Mixed farming communities with established coping mechanisms.",
    watchItems: ["Crop failure indicators"],
    sourceRegionId: "placeholder",
    population: "580,000",
  },
  {
    id: "central_highlands",
    shapeId: "central_highlands",
    name: "Central Highlands",
    riskLevel: "low",
    riskScore: 0.18,
    thresholdStatus: "Below threshold",
    primaryDriver: "Political sensitivity",
    secondaryDriver: "Election period",
    confidence: "High",
    storySummary: "Stable conditions with commercial agriculture performing normally.",
    thresholdReason: "Rainfall and economic indicators within expected range.",
    communityContext: "Commercial agriculture, strong market linkages.",
    watchItems: ["Election-period tension indicators"],
    sourceRegionId: "placeholder",
    population: "1,200,000",
  },
  {
    id: "nairobi_metro",
    shapeId: "nairobi_metro",
    name: "Nairobi Metropolitan",
    riskLevel: "low",
    riskScore: 0.22,
    thresholdStatus: "Below threshold",
    primaryDriver: "Urban unrest risk",
    secondaryDriver: "Economic inequality",
    confidence: "High",
    storySummary: "Stable. Monitor urban cost-of-living triggers.",
    thresholdReason: "Urban risk signals below threshold.",
    communityContext: "High-density urban population, service economy.",
    watchItems: ["Cost of living index", "Urban unemployment rates"],
    sourceRegionId: "placeholder",
    population: "4,400,000",
  },
  {
    id: "coast_belt",
    shapeId: "coast_belt",
    name: "Coast Belt",
    riskLevel: "watch",
    riskScore: 0.44,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Radicalization corridors",
    secondaryDriver: "Economic marginalization",
    confidence: "Medium",
    storySummary: "Long-standing risk corridors under monitoring.",
    thresholdReason: "Radicalization indicators slightly elevated.",
    communityContext: "Tourism, trade, and fishing communities. Historical marginalization.",
    watchItems: ["Security incident frequency", "Youth unemployment indicators"],
    sourceRegionId: "placeholder",
    population: "1,100,000",
  },
  {
    id: "south_rift",
    shapeId: "south_rift",
    name: "South Rift Valley",
    riskLevel: "watch",
    riskScore: 0.41,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Livestock theft",
    secondaryDriver: "Land boundary disputes",
    confidence: "Medium",
    storySummary: "Seasonal tension elevation in pastoral zones.",
    thresholdReason: "Cross-community livestock theft slightly elevated.",
    communityContext: "Pastoral and small-holder communities with inter-ethnic friction points.",
    watchItems: ["Livestock theft incident rate", "Mediation requests"],
    sourceRegionId: "placeholder",
    population: "630,000",
  },
];
