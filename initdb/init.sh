#!/usr/bin/env bash
set -euo pipefail

EMBEDDING_DIM="${EMBEDDING_DIM:-768}"

echo "Running idempotent DB setup with embedding_dim=${EMBEDDING_DIM}..."
sed "s/:embedding_dim/${EMBEDDING_DIM}/g" /docker-entrypoint-initdb.d/setup.sql.in \
  | psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"
echo "DB setup complete."
