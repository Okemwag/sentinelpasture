"use client";

interface PanelProps {
  title?: string;
  children: React.ReactNode;
  metadata?: {
    modelVersion?: string;
    lastUpdated?: string;
    confidence?: string;
    showExplainLink?: boolean;
  };
}

export default function Panel({ title, children, metadata }: PanelProps) {
  const handleExplain = () => {
    alert(
      "Methodology explanation:\n\n" +
      "- Data sources: monthly rainfall feature snapshots plus event-label histories\n" +
      "- Model: baseline-risk-model-v1 (correlation-weighted statistical baseline)\n" +
      "- Scoring: min-max normalized features weighted by historical relationship to event and fatality pressure\n" +
      "- Thresholds: low / watch / elevated / critical bands from current risk score\n" +
      "- Confidence: observation-count heuristic\n" +
      "- Narratives: region-specific operational templates tied to the top driver\n" +
      "- Interventions: rule-based options constrained by policy guardrails"
    );
  };

  return (
    <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-6">
      {title && (
        <h2 className="text-[18px] font-medium text-[#111111] mb-4">
          {title}
        </h2>
      )}
      
      {children}
      
      {metadata && (
        <div className="mt-6 pt-4 border-t border-[#E5E7EB]">
          <div className="flex flex-wrap gap-4 text-[12px] text-[#6B7280]">
            {metadata.modelVersion && (
              <span>Model: {metadata.modelVersion}</span>
            )}
            {metadata.lastUpdated && (
              <span>Updated: {metadata.lastUpdated}</span>
            )}
            {metadata.confidence && (
              <span>Confidence: {metadata.confidence}</span>
            )}
            {metadata.showExplainLink && (
              <button
                onClick={handleExplain}
                className="text-[#374151] underline hover:no-underline"
              >
                How was this computed?
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
