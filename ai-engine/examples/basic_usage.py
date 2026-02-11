"""
Basic Usage Example for AI Engine
"""

import asyncio
from datetime import datetime

from ai_engine import AIOrchestrator, Settings, setup_logging


async def main():
    """Basic usage example"""
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Create settings
    settings = Settings()
    settings.model.device = "cpu"
    settings.debug = True
    
    # Create orchestrator
    orchestrator = AIOrchestrator(settings=settings)
    
    # Create sample signals
    raw_signals = [
        {
            "type": "social_media",
            "source": "verified_media",
            "data": {
                "text": "Large gathering reported in downtown area",
                "likes": 5000,
                "shares": 1200,
                "verified": True
            },
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "region": "New York"
            },
            "temporal": {
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "type": "economic",
            "source": "official_government",
            "data": {
                "gdp": 21000,
                "inflation": 3.2,
                "unemployment": 4.5
            },
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "region": "New York"
            },
            "temporal": {
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    
    # Process signals through pipeline
    print("Processing signals...")
    result = await orchestrator.process_intelligence_pipeline(raw_signals)
    
    # Display results
    print("\n=== Assessment ===")
    assessment = result["assessment"]
    print(f"Threat Level: {assessment['threat_level']}")
    print(f"Confidence: {assessment['confidence']:.2%}")
    print(f"Pressure Index: {assessment['pressure_index']['overall']:.2%}")
    
    print("\n=== Indicators ===")
    for i, indicator in enumerate(result["indicators"][:3], 1):
        print(f"{i}. {indicator['name']} (Severity: {indicator['severity']})")
    
    print("\n=== Key Insights ===")
    insights = result["insights"]
    for finding in insights["key_findings"][:3]:
        print(f"- {finding}")
    
    print("\n=== System Status ===")
    status = orchestrator.get_system_status()
    print(f"Status: {status['status']}")
    print(f"Signals Processed: {status['performance']['signals_processed']}")
    
    # Export report
    report = await orchestrator.export_assessment_report(assessment["id"])
    print(f"\n=== Report Generated ===")
    print(f"Assessment ID: {assessment['id']}")
    print(f"Exported at: {report['metadata']['exported_at']}")


if __name__ == "__main__":
    asyncio.run(main())
