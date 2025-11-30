#!/bin/bash
#
# Cleanup .git directories from cloned repositories
# This prevents nested git repo issues when pushing to GitHub
#

set -e

CODE_REPOS_DIR="data/raw/code_repos"

if [ ! -d "$CODE_REPOS_DIR" ]; then
    echo "Directory $CODE_REPOS_DIR does not exist"
    exit 1
fi

echo "Removing .git directories from $CODE_REPOS_DIR..."
echo "This prevents nested git repo issues when pushing to GitHub"
echo ""

# Count .git directories
GIT_COUNT=$(find "$CODE_REPOS_DIR" -name ".git" -type d | wc -l | tr -d ' ')
echo "Found $GIT_COUNT .git directories"

if [ "$GIT_COUNT" -eq 0 ]; then
    echo "No .git directories to remove"
    exit 0
fi

# Remove .git directories
find "$CODE_REPOS_DIR" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# Count remaining
REMAINING=$(find "$CODE_REPOS_DIR" -name ".git" -type d 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "Cleanup complete!"
echo "Removed: $((GIT_COUNT - REMAINING)) .git directories"
echo "Remaining: $REMAINING .git directories"

if [ "$REMAINING" -gt 0 ]; then
    echo ""
    echo "Note: Some .git directories couldn't be removed (permissions?)"
    echo "Run with sudo if needed"
fi
