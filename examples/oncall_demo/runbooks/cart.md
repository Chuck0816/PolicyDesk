---
doc_type: runbook
service: cart
title: Cart runbook
---
# Cart runbook

## Cache connection pool exhausted
Alert: CartCacheConnectionsHigh. valkey-cart connected clients above 900.

1. Check `valkey-cli info clients` on valkey-cart.
2. Look for a recent cart deploy that changed the pool size or leaked connections.
3. Restart cart pods to release connections: `kubectl rollout restart deploy/cart`.
```bash
# do not flush the cache; carts would be lost
kubectl rollout restart deploy/cart
```
4. If a deploy caused it, roll it back. Owner: team-checkout.

## Cart data missing
valkey-cart is a cache without persistence. A valkey-cart restart empties all carts; tell support before restarting it.
