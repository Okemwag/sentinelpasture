"""
Database initialization script
Creates tables and seeds initial data
"""

import asyncio
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from database import init_db, AsyncSessionLocal
from models import User, UserRole, RegionalData, Alert
from auth import get_password_hash


async def create_initial_users():
    """Create initial admin and test users"""
    async with AsyncSessionLocal() as session:
        # Create admin user
        admin = User(
            email="admin@riskplatform.gov",
            username="admin",
            hashed_password=get_password_hash("admin123"),
            full_name="System Administrator",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        session.add(admin)
        
        # Create analyst user
        analyst = User(
            email="analyst@riskplatform.gov",
            username="analyst",
            hashed_password=get_password_hash("analyst123"),
            full_name="Risk Analyst",
            role=UserRole.ANALYST,
            is_active=True,
            is_verified=True
        )
        session.add(analyst)
        
        # Create viewer user
        viewer = User(
            email="viewer@riskplatform.gov",
            username="viewer",
            hashed_password=get_password_hash("viewer123"),
            full_name="Data Viewer",
            role=UserRole.VIEWER,
            is_active=True,
            is_verified=True
        )
        session.add(viewer)
        
        await session.commit()
        print("✅ Initial users created")
        print("   - admin / admin123 (Admin)")
        print("   - analyst / analyst123 (Analyst)")
        print("   - viewer / viewer123 (Viewer)")


async def create_sample_regional_data():
    """Create sample regional data"""
    async with AsyncSessionLocal() as session:
        regions = [
            {
                "region_code": "KE-01",
                "region_name": "Nairobi",
                "risk_level": 45.2,
                "stability_index": 72.5,
                "trend": "stable"
            },
            {
                "region_code": "KE-02",
                "region_name": "Mombasa",
                "risk_level": 52.8,
                "stability_index": 68.3,
                "trend": "increasing"
            },
            {
                "region_code": "KE-03",
                "region_name": "Kisumu",
                "risk_level": 38.5,
                "stability_index": 75.2,
                "trend": "stable"
            },
            {
                "region_code": "KE-04",
                "region_name": "Nakuru",
                "risk_level": 41.3,
                "stability_index": 73.8,
                "trend": "decreasing"
            }
        ]
        
        for region_data in regions:
            regional = RegionalData(
                **region_data,
                data_date=datetime.utcnow(),
                metrics={
                    "economic_stress": 0.45,
                    "social_tension": 0.32,
                    "environmental_risk": 0.28
                },
                drivers=[
                    {"name": "Economic stress", "value": 0.45},
                    {"name": "Social tension", "value": 0.32}
                ],
                alerts=[]
            )
            session.add(regional)
        
        await session.commit()
        print("✅ Sample regional data created")


async def create_sample_alerts():
    """Create sample alerts"""
    async with AsyncSessionLocal() as session:
        alerts = [
            Alert(
                alert_type="economic",
                severity="high",
                title="Rising unemployment in urban areas",
                message="Unemployment rate has increased by 3.2% in major urban centers",
                region="KE-01",
                source="Economic Monitoring System",
                status="active",
                metadata={"unemployment_rate": 15.2, "change": 3.2}
            ),
            Alert(
                alert_type="environmental",
                severity="medium",
                title="Drought conditions detected",
                message="Below-average rainfall in agricultural regions",
                region="KE-03",
                source="Climate Monitoring System",
                status="active",
                metadata={"rainfall_deficit": 35}
            )
        ]
        
        for alert in alerts:
            session.add(alert)
        
        await session.commit()
        print("✅ Sample alerts created")


async def main():
    """Main initialization function"""
    print("🚀 Initializing database...")
    
    try:
        # Create tables
        await init_db()
        print("✅ Database tables created")
        
        # Seed data
        await create_initial_users()
        await create_sample_regional_data()
        await create_sample_alerts()
        
        print("\n✅ Database initialization complete!")
        print("\n📝 You can now start the server with: python main_enhanced.py")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
