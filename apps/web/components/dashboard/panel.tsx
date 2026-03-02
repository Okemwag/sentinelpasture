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
    // In production, this would open a modal with detailed methodology
    alert("Methodology explanation:\n\nThis metric is computed using:\n- Data sources: National statistical agencies, sensor networks\n- Model: Ensemble prediction system v2.4.1\n- Confidence calculation: Based on data quality and model agreement\n- Last calibration: 2026-02-15\n\nFor detailed documentation, see the technical reference manual.");
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
