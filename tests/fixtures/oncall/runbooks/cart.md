---
service: cart
---
# Cart runbook
## Cache connection pool exhausted
Check valkey-cart connected clients. Restart cart pods if the pool is stuck.
```bash
# not a heading inside code
kubectl rollout restart deploy/cart
```
