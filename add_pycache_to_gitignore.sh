#!/bin/bash

# Find all __pycache__ directories and append them to pycache_paths.txt

find . -type d -name "__pycache__" -exec echo "{}" \; | sed 's|^\./||' > pycache_paths.txt

echo "All __pycache__ directories added to pycache_paths.txt."
