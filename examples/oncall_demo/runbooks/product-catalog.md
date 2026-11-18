---
doc_type: runbook
service: product-catalog
title: Product catalog runbook
---
# Product catalog runbook

## Database lock contention
Alert: CatalogDbSlowQueries. Many queries waiting on locks in postgresql.

1. Find blocking queries: `SELECT pid, query FROM pg_stat_activity WHERE wait_event_type = 'Lock';`
2. Long-running migrations or batch price updates are the usual cause. Cancel the blocker with `pg_cancel_backend(pid)`.
3. Owner: team-catalog. Frontend and recommendation slow down while the catalog is slow.

## Search returns no results
Rebuild the search index: `kubectl exec deploy/product-catalog -- ./reindex`.
