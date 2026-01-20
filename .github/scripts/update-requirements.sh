#!/usr/bin/env bash
#
# Update requirements.txt from pyproject.toml using uv pip compile.
# Used by pre-commit hook to keep requirements.txt in sync.
#

set -euo pipefail

# Dependency groups to include in requirements.txt
DEPENDENCY_GROUPS=(
    "dev"
    "text"
    "visualization"
    "workshop"
)

# Build the group arguments
GROUP_ARGS=""
for group in "${DEPENDENCY_GROUPS[@]}"; do
    GROUP_ARGS="$GROUP_ARGS --group $group"
done

# Compile requirements
uv pip compile pyproject.toml -o requirements.txt --no-deps $GROUP_ARGS

# Auto-stage if running in git context (pre-commit hook)
if git rev-parse --git-dir > /dev/null 2>&1; then
    git add requirements.txt
fi
