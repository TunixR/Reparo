# Upgrading the Docker stack

Manual steps needed when moving between versions of the `docker/` structure,
newest first. Version bumps that need no manual work (image pins, config
tweaks) are not listed. See [README.md](README.md) → Upgrades for routine
image upgrades.

## v2: R2Framework → Reparo

The project was renamed. Every Docker name and default moved from `r2framework`
(and the `gateway` database and bucket) to `reparo`:

- Compose project `r2framework` → `reparo`; volumes `r2framework-*` →
  `reparo-*`; networks `r2framework-backend` / `-telemetry` → `reparo-*`;
  image `r2framework:<tag>` → `reparo:<tag>`.
- `POSTGRES_DB` and `S3_BUCKET` defaults `gateway` → `reparo`.
- `service.name` `r2framework-api` → `reparo-api`, `service.namespace`
  `r2framework` → `reparo`.

Without these steps the new stack starts empty next to the old one. The steps
below keep the data and rename the database and bucket; to skip that, keep
`POSTGRES_DB=gateway` (both `.env` files) and `S3_BUCKET=gateway` and leave out
steps 4 and 6. Use `-f docker/compose.prod.yaml` in production.

1. Stop and remove the old containers and networks. The compose files now name
   the project `reparo`, so `docker compose down` no longer finds them; stop
   them by label, with time for a clean Postgres shutdown:
   ```bash
   ids=$(docker ps -aq --filter label=com.docker.compose.project=r2framework)
   docker stop -t 60 $ids && docker rm $ids
   docker network rm r2framework-backend r2framework-telemetry
   ```
2. Copy every volume into its new name (Postgres, SeaweedFS and SigNoz data,
   including the SigNoz account). Any image with `sh` and `cp` works:
   ```bash
   for v in $(docker volume ls -q | grep '^r2framework-'); do
     n="reparo-${v#r2framework-}"
     docker volume create "$n"
     docker run --rm -v "$v":/from:ro -v "$n":/to alpine sh -c 'cp -a /from/. /to/'
   done
   ```
3. Update the settings: `POSTGRES_DB=reparo` in `docker/.env` and `.env`;
   `S3_BUCKET`, `OTEL_SERVICE_NAME`, `OTEL_SERVICE_NAMESPACE` (and any
   `OTEL_RESOURCE_ATTRIBUTES`, `SIGNOZ_CLICKHOUSE_CONTAINER`) in `.env`.
4. Start Postgres alone and rename the database before anything else connects
   to it (`POSTGRES_DB` only applies to an empty volume):
   ```bash
   docker compose -f docker/compose.dev.yaml up -d --wait postgres
   docker exec reparo-postgres-1 sh -c \
     'psql -U "$POSTGRES_USER" -d postgres -c "ALTER DATABASE gateway RENAME TO reparo"'
   ```
5. Start the rest: `docker compose -f docker/compose.dev.yaml up -d --wait`
   (production: rebuild the image first). A healthy `signoz-collector` means
   the SigNoz account came across.
6. Copy the bucket. Screenshot rows store only the object key, so old
   screenshots resolve once the objects are in `reparo`. With the AWS CLI:
   `aws s3 sync s3://gateway s3://reparo --endpoint-url http://localhost:8333`
   (or `copy_object` per key with boto3).
7. Run `uv sync` and `uv run alembic current`. Old telemetry stays under the old
   service names: edit SigNoz views, dashboards and alerts that filter on them.
8. Once the data checks out, delete the `r2framework-*` volumes and the
   `gateway` bucket.

## v1

Initial version of this structure. Nothing to upgrade from.
