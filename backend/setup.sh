#!/bin/bash

# Enhanced Backend Setup Script
# Sets up PostgreSQL, Python environment, and initializes database

set -e

echo "🚀 Setting up Enhanced FastAPI Backend..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if PostgreSQL is installed
echo "📦 Checking PostgreSQL..."
if ! command -v psql &> /dev/null; then
    echo -e "${RED}❌ PostgreSQL is not installed${NC}"
    echo ""
    echo "Please install PostgreSQL:"
    echo "  macOS:   brew install postgresql@15"
    echo "  Ubuntu:  sudo apt-get install postgresql postgresql-contrib"
    echo "  Windows: Download from https://www.postgresql.org/download/"
    exit 1
fi

echo -e "${GREEN}✅ PostgreSQL found${NC}"

# Check if PostgreSQL is running
if ! pg_isready &> /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL is not running${NC}"
    echo "Starting PostgreSQL..."
    
    # Try to start PostgreSQL (platform-specific)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql@15
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
    fi
    
    sleep 2
    
    if ! pg_isready &> /dev/null; then
        echo -e "${RED}❌ Could not start PostgreSQL${NC}"
        echo "Please start PostgreSQL manually"
        exit 1
    fi
fi

echo -e "${GREEN}✅ PostgreSQL is running${NC}"

# Create database if it doesn't exist
echo ""
echo "📊 Setting up database..."
if psql -U postgres -lqt | cut -d \| -f 1 | grep -qw risk_intelligence; then
    echo -e "${YELLOW}⚠️  Database 'risk_intelligence' already exists${NC}"
    read -p "Do you want to drop and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        dropdb -U postgres risk_intelligence
        createdb -U postgres risk_intelligence
        echo -e "${GREEN}✅ Database recreated${NC}"
    fi
else
    createdb -U postgres risk_intelligence
    echo -e "${GREEN}✅ Database created${NC}"
fi

# Setup Python virtual environment
echo ""
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Setup environment file
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    
    # Generate secret key
    SECRET_KEY=$(openssl rand -hex 32)
    
    # Update .env with generated secret key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/your-secret-key-change-in-production-use-openssl-rand-hex-32/$SECRET_KEY/" .env
    else
        sed -i "s/your-secret-key-change-in-production-use-openssl-rand-hex-32/$SECRET_KEY/" .env
    fi
    
    echo -e "${GREEN}✅ Environment file created with secure secret key${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists${NC}"
fi

# Initialize database
echo ""
echo "🗄️  Initializing database..."
python init_db.py

# Success message
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Backend setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📝 Default users created:"
echo "   • admin / admin123 (Admin)"
echo "   • analyst / analyst123 (Analyst)"
echo "   • viewer / viewer123 (Viewer)"
echo ""
echo "🚀 Start the server:"
echo "   python main_enhanced.py"
echo ""
echo "📚 API Documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "🔗 WebSocket:"
echo "   ws://localhost:8000/ws/default"
echo ""
echo -e "${YELLOW}⚠️  Remember to change default passwords in production!${NC}"
echo ""
