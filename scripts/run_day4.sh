#!/bin/bash
# Day 4: Data Preparation Pipeline
# Runs all Day 4 steps to prepare data for contrastive learning training

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Day 4: Data Preparation Pipeline"
echo "==========================================${NC}"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Step 2: Parse paper-code pairs
echo ""
echo -e "${BLUE}Step 2: Parsing paper_code_pairs.json...${NC}"
python src/embeddings/paper_code_parser.py

if [ ! -f "data/processed/parsed_pairs.json" ]; then
    echo -e "${YELLOW}Warning: parsed_pairs.json was not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 2 complete${NC}"

# Step 4: Create DataLoaders and save info
echo ""
echo -e "${BLUE}Step 4: Setting up DataLoaders...${NC}"
python src/embeddings/data_loader_setup.py

if [ ! -f "data/processed/dataset_info.json" ]; then
    echo -e "${YELLOW}Warning: dataset_info.json was not created${NC}"
fi

echo -e "${GREEN}✓ Step 4 complete${NC}"

# Summary
echo ""
echo -e "${GREEN}Files created:${NC}"
echo "  - data/processed/parsed_pairs.json"
if [ -f "data/processed/dataset_info.json" ]; then
    echo "  - data/processed/dataset_info.json"
fi
echo ""

