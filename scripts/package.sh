#!/usr/bin/env bash
# Build the submission zip from an allowlist, so secrets and local artifacts can't slip in.
#   scripts/package.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-../claims-sop-agent.zip}"
NAME="insurance_claims"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/$NAME"

INCLUDE=(app static fixtures tests scripts docs
         README.md DESIGN.md Dockerfile docker-compose.yml .dockerignore .gitignore .env.example
         requirements.txt requirements-dev.txt pytest.ini)
for p in "${INCLUDE[@]}"; do
  cp -R "$p" "$STAGE/$NAME/"
done
find "$STAGE" \( -name '__pycache__' -o -name '.claude' -o -name '.pytest_cache' -o -name '.DS_Store' \) -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete

# hard stops: nothing secret or generated may be in the bundle
if find "$STAGE" -name '.env' -o -name '.venv' -o -name 'outbox' -o -name 'logs' | grep -q .; then
  echo "refusing to package: secret or generated files found" >&2; exit 1
fi
if grep -rIlE 'sk-[A-Za-z0-9_-]{32,}' "$STAGE" >/dev/null; then
  echo "refusing to package: something that looks like an API key is in the files" >&2; exit 1
fi

rm -f "$OUT"
(cd "$STAGE" && zip -qr - "$NAME") > "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files"
