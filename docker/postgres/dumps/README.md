# Postgres seed dumps (development)

Files here are git- and docker-ignored. When the dev Postgres starts with an
empty data volume, `initdb/20-restore-dumps.sh` restores each one into the
database named after the file (`reparo.dump` → `reparo`).

Supported: `*.dump` (`pg_dump -Fc`), `*.sql`, `*.sql.gz`.

```bash
pg_dump -h localhost -U postgres -d reparo -Fc -f docker/postgres/dumps/reparo.dump
```

To re-seed, recreate the volume (this deletes the current dev database):

```bash
docker compose -f docker/compose.dev.yaml rm -sf postgres
docker volume rm reparo-postgres-data
docker compose -f docker/compose.dev.yaml up -d postgres
```
