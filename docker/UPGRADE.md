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

Without these steps the new stack starts empty next to the old one.

1. Stop the old stack: `docker compose -p r2framework down`.
2. Copy each volume into its new name, for every `r2framework-*` volume:
   ```bash
   docker volume create reparo-postgres-data
   docker run --rm -v r2framework-postgres-data:/from -v reparo-postgres-data:/to \
     alpine sh -c 'cp -a /from/. /to/'
   ```
3. Either keep the old names by setting `POSTGRES_DB=gateway` (both `.env` files)
   and `S3_BUCKET=gateway`, or rename them: start Postgres, run
   `ALTER DATABASE gateway RENAME TO reparo;` from the `postgres` database, and
   copy the bucket (`aws s3 sync s3://gateway s3://reparo --endpoint-url http://localhost:8333`).
4. Update `OTEL_SERVICE_NAME`, `OTEL_SERVICE_NAMESPACE` (and any
   `OTEL_RESOURCE_ATTRIBUTES`, `SIGNOZ_CLICKHOUSE_CONTAINER`) in `.env`. Old
   telemetry stays under the old names: edit SigNoz views, dashboards and alerts
   that filter on them.
5. Rebuild the image (production), `uv sync`, start the stack, then remove the
   `r2framework-*` volumes once the data checks out.

## v1

Initial version of this structure. Nothing to upgrade from.
