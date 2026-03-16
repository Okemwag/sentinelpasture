"""Kenya regional storytelling helpers for dashboard-ready risk views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegionProfile:
    id: str
    name: str
    shape_id: str
    population: str
    exposure: str
    community_context: str
    base_watch_items: tuple[str, str, str]


REGION_PROFILES: tuple[RegionProfile, ...] = (
    RegionProfile(
        id="north_west_frontier",
        name="North Western Frontier",
        shape_id="north_west_frontier",
        population="Pastoral and agro-pastoral households",
        exposure="Livestock corridors, water access, and dryland roads",
        community_context=(
            "Mobility decisions here often depend on pasture continuity, security conditions, "
            "and whether livestock routes remain open to key markets and boreholes."
        ),
        base_watch_items=(
            "Water point congestion and trucking requests",
            "Livestock condition reports and market arrivals",
            "Access constraints on dryland feeder roads",
        ),
    ),
    RegionProfile(
        id="north_eastern_drylands",
        name="North Eastern Drylands",
        shape_id="north_eastern_drylands",
        population="Pastoral households and cross-border trading communities",
        exposure="Pasture shocks, herd movement, and borderland market timing",
        community_context=(
            "Many communities across this belt are pastoral and Cushitic-speaking, so abrupt "
            "rainfall shifts can change grazing calendars, livestock market timing, and "
            "cross-border support routes at the same time."
        ),
        base_watch_items=(
            "Pasture concentration around a small number of corridors",
            "Livestock price swings and distress sales",
            "Water-point pressure, borehole outages, and local mediation requests",
        ),
    ),
    RegionProfile(
        id="upper_eastern_corridor",
        name="Upper Eastern Corridor",
        shape_id="upper_eastern_corridor",
        population="Mixed dryland farming and market-linked households",
        exposure="Rain-fed agriculture, transport corridors, and grazing spillover",
        community_context=(
            "Risk here tends to move through the interaction between dryland wards, market "
            "towns, and transport links that connect pastoral and farming systems."
        ),
        base_watch_items=(
            "Rain-fed crop stress and late planting signals",
            "Pasture spillover into mixed-farming belts",
            "Road interruptions along market corridors",
        ),
    ),
    RegionProfile(
        id="lake_basin",
        name="Lake Basin",
        shape_id="lake_basin",
        population="Dense rural households and urban market centers",
        exposure="Floodplains, food transport, and market volatility",
        community_context=(
            "The basin is sensitive to short-horizon weather shifts because logistics, "
            "food prices, and floodplain access all move together."
        ),
        base_watch_items=(
            "Flood-prone roads and bridge access",
            "Food basket price movement in local markets",
            "Displacement pressure near low-lying settlements",
        ),
    ),
    RegionProfile(
        id="central_highlands",
        name="Central Highlands",
        shape_id="central_highlands",
        population="High-density farming and trading households",
        exposure="Food basket supply, feeder roads, and peri-urban demand",
        community_context=(
            "Threshold breaches here often matter nationally because this belt influences "
            "food supply, transport reliability, and urban market expectations."
        ),
        base_watch_items=(
            "Feeder-road delays from farms to collection points",
            "Wholesale food-price movement into Nairobi",
            "Input shortages or delayed harvest movement",
        ),
    ),
    RegionProfile(
        id="nairobi_metro",
        name="Nairobi Metro",
        shape_id="nairobi_metro",
        population="Urban households, commuters, and informal-settlement communities",
        exposure="Price sensitivity, rumor spread, and service strain",
        community_context=(
            "Pressure in the capital can escalate quickly when food prices, flooding, "
            "transport disruption, and political tension reinforce one another."
        ),
        base_watch_items=(
            "Commuter disruption and flooding hotspots",
            "Retail food-price spikes in dense neighborhoods",
            "Local service stress and incident chatter",
        ),
    ),
    RegionProfile(
        id="coast_belt",
        name="Coast Belt",
        shape_id="coast_belt",
        population="Port-linked urban centers and coastal livelihoods",
        exposure="Flooding, logistics access, and tourism-linked income shocks",
        community_context=(
            "The coastal belt is sensitive to rainfall pattern shifts because transport, "
            "port access, and peri-urban flooding pressures move together."
        ),
        base_watch_items=(
            "Port and corridor throughput disruption",
            "Peri-urban flooding and drainage failures",
            "Price transmission into coastal urban markets",
        ),
    ),
    RegionProfile(
        id="south_rift",
        name="South Rift",
        shape_id="south_rift",
        population="Mixed farming, grazing, and corridor-linked communities",
        exposure="Mixed-livelihood pressure, land access, and inter-county movement",
        community_context=(
            "The south rift tends to show early warning through mixed-livelihood stress: "
            "farm output, grazing pressure, and corridor security all matter at once."
        ),
        base_watch_items=(
            "Grazing pressure near settlement edges",
            "Farm-to-market movement and local price spreads",
            "Transport chokepoints and localized displacement reports",
        ),
    ),
)

RAIN_DRIVERS = {"Rainfall anomaly", "Rainfall level", "Rainfall baseline", "Quarterly rainfall shift"}
PRESSURE_DRIVERS = {"One-month pressure", "Three-month pressure"}


def build_regional_briefing(rows: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    profiles = REGION_PROFILES[:limit] if limit is not None else REGION_PROFILES
    ordered_rows = _ordered_rows(rows)

    briefing: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        source_row = ordered_rows[index % len(ordered_rows)]
        risk_score = round(float(source_row.get("risk_score", 0.0)), 4)
        confidence_value = round(float(source_row.get("confidence", 0.0)), 2)
        driver = str(source_row.get("primary_driver", "Model-derived pressure"))
        risk_level = _risk_level(risk_score)
        threshold_status = _threshold_status(risk_level)
        confidence_label = _confidence_label(confidence_value)

        briefing.append(
            {
                "id": profile.id,
                "shapeId": profile.shape_id,
                "name": profile.name,
                "region": profile.name,
                "population": profile.population,
                "exposure": profile.exposure,
                "communityContext": profile.community_context,
                "sourceRegionId": str(source_row.get("region_id", f"synthetic-{index + 1}")),
                "riskScore": risk_score,
                "stabilityIndex": max(0, int(round(100 - (risk_score * 100)))),
                "riskLevel": risk_level,
                "thresholdStatus": threshold_status,
                "trend": threshold_status,
                "primaryDriver": driver,
                "secondaryDriver": profile.exposure,
                "confidence": confidence_label,
                "storySummary": _story_summary(profile, driver, risk_level),
                "thresholdReason": _threshold_reason(profile, driver, risk_level),
                "watchItems": _watch_items(profile, driver),
                "featureSnapshotTimestamp": source_row.get("feature_snapshot_timestamp", ""),
            }
        )

    return briefing


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows:
        return sorted(rows, key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    return [
        {
            "region_id": f"synthetic-{index + 1}",
            "risk_score": max(0.32, 0.78 - (index * 0.05)),
            "confidence": 0.77,
            "primary_driver": "Rainfall anomaly" if index < 3 else "Three-month pressure",
            "feature_snapshot_timestamp": "",
        }
        for index in range(len(REGION_PROFILES))
    ]


def _risk_level(score: float) -> str:
    if score >= 0.76:
        return "critical"
    if score >= 0.61:
        return "elevated"
    if score >= 0.46:
        return "watch"
    return "low"


def _threshold_status(risk_level: str) -> str:
    if risk_level == "critical":
        return "Critical escalation"
    if risk_level == "elevated":
        return "Threshold crossed"
    if risk_level == "watch":
        return "Watch threshold"
    return "Below threshold"


def _confidence_label(value: float) -> str:
    return "High" if value >= 0.8 else "Medium" if value >= 0.65 else "Low"


def _story_summary(profile: RegionProfile, driver: str, risk_level: str) -> str:
    if profile.id == "north_eastern_drylands" and driver in RAIN_DRIVERS:
        return (
            "Sudden rainfall shifts in the north eastern drylands can green pasture unevenly, "
            "pulling herds toward a smaller number of water points and grazing corridors before "
            "local services rebalance."
        )
    if driver in RAIN_DRIVERS:
        return (
            f"{profile.name} is sensitive to abrupt rainfall shifts because they change access to "
            f"{profile.exposure.lower()} faster than households, markets, and local authorities can adjust."
        )
    if driver in PRESSURE_DRIVERS:
        return (
            f"{profile.name} is showing a cumulative stress pattern rather than a one-off shock. "
            "That usually means the threshold is being pushed by repeated strain across several reporting windows."
        )
    if risk_level in {"critical", "elevated"}:
        return (
            f"{profile.name} is above its watch band because current drivers are clustering in the same "
            "operational corridor, increasing the chance of rapid spillover."
        )
    return f"{profile.name} remains below the main alert threshold but should be monitored for early movement."


def _threshold_reason(profile: RegionProfile, driver: str, risk_level: str) -> str:
    if profile.id == "north_eastern_drylands" and driver in RAIN_DRIVERS:
        return (
            "When rainfall patterns break sharply from the local norm, pasture and water access stop being evenly "
            "distributed. That can shift herd movement, livestock market timing, and local coexistence pressure very quickly."
        )
    if driver in RAIN_DRIVERS:
        return (
            f"The current threshold matters because rainfall-driven shocks in {profile.name.lower()} affect "
            "livelihood access before they show up cleanly in incident counts."
        )
    if driver in PRESSURE_DRIVERS:
        return (
            "The threshold is being crossed by repeated pressure, not a single signal. That is risky because service strain, "
            "movement pressure, and local tension can compound before any one indicator looks extreme on its own."
        )
    if risk_level == "critical":
        return "Several risk drivers are now aligned. That makes local disruption more likely to spread into nearby markets, roads, and service points."
    return "The alert threshold is being approached because local stress is no longer isolated to one signal or one time window."


def _watch_items(profile: RegionProfile, driver: str) -> list[str]:
    if driver in RAIN_DRIVERS:
        driver_watch = "Short-horizon rainfall deviation and pasture concentration"
    elif driver in PRESSURE_DRIVERS:
        driver_watch = "Repeated stress accumulation across consecutive reporting windows"
    else:
        driver_watch = "Local signals that could push the region into a higher alert band"
    return [driver_watch, *profile.base_watch_items[:2]]
