export type RegionalRiskLevel = "low" | "watch" | "elevated" | "critical";

export const KENYA_OUTLINE_PATH =
  "M132 34 L184 24 L238 36 L286 70 L318 130 L324 188 L314 248 L330 324 L312 406 L272 484 L226 530 L176 522 L138 474 L118 418 L112 344 L98 274 L92 204 L98 138 L114 78 Z";

export const KENYA_REGION_SHAPES: Record<
  string,
  { points: string; label: string; labelX: number; labelY: number; fullName: string }
> = {
  north_west_frontier: {
    points: "114,86 152,52 192,70 186,126 154,178 116,166 100,126",
    label: "NW",
    labelX: 146,
    labelY: 116,
    fullName: "North West Frontier",
  },
  north_eastern_drylands: {
    points: "190,70 246,56 296,94 312,166 286,214 236,204 204,168 184,124",
    label: "NE",
    labelX: 250,
    labelY: 128,
    fullName: "North Eastern Drylands",
  },
  upper_eastern_corridor: {
    points: "154,178 184,124 204,168 236,204 232,260 188,294 148,246",
    label: "UE",
    labelX: 193,
    labelY: 218,
    fullName: "Upper Eastern Corridor",
  },
  lake_basin: {
    points: "100,166 154,178 148,246 118,320 98,356 78,290 84,226",
    label: "LB",
    labelX: 115,
    labelY: 260,
    fullName: "Lake Basin Region",
  },
  central_highlands: {
    points: "148,246 188,294 208,346 180,390 134,374 118,320",
    label: "CH",
    labelX: 164,
    labelY: 325,
    fullName: "Central Highlands",
  },
  nairobi_metro: {
    points: "186,392 206,386 214,410 198,430 178,424",
    label: "NM",
    labelX: 197,
    labelY: 409,
    fullName: "Nairobi Metropolitan",
  },
  coast_belt: {
    points: "208,346 256,330 286,380 282,460 232,516 182,492 178,438 196,430 214,410",
    label: "CB",
    labelX: 244,
    labelY: 414,
    fullName: "Coast Belt",
  },
  south_rift: {
    points: "134,374 180,390 178,438 182,492 144,486 108,442 100,390",
    label: "SR",
    labelX: 145,
    labelY: 438,
    fullName: "South Rift Valley",
  },
};

export function getRiskPalette(level: RegionalRiskLevel) {
  switch (level) {
    case "critical":
      return {
        fill: "#4A3A26",
        stroke: "#2A1E12",
        badge: "bg-[#EDE8E0] text-[#2A1E12] border-[#BFB099]",
        label: "Critical",
        textColor: "#2A1E12",
      };
    case "elevated":
      return {
        fill: "#8C6A3D",
        stroke: "#5A3E1E",
        badge: "bg-[#F5EFE6] text-[#5A3E1E] border-[#D4B98A]",
        label: "High",
        textColor: "#5A3E1E",
      };
    case "watch":
      return {
        fill: "#C7A56B",
        stroke: "#8C6A3D",
        badge: "bg-[#FBF5EB] text-[#7A4F1E] border-[#E2C99A]",
        label: "Elevated",
        textColor: "#7A4F1E",
      };
    default:
      return {
        fill: "#C8D8C2",
        stroke: "#7A9B70",
        badge: "bg-[#EDF4EB] text-[#3A6B33] border-[#A8C9A2]",
        label: "Low",
        textColor: "#3A6B33",
      };
  }
}

export function getRiskLevelLabel(level: RegionalRiskLevel): string {
  return getRiskPalette(level).label;
}
