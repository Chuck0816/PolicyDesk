---
doc_type: runbook
service: checkout
title: Checkout runbook
---
# Checkout runbook

## Latency high
Alert: CheckoutLatencyHigh. PlaceOrder p99 above 800 ms.

1. Open the checkout dashboard and check which downstream call is slow: cart, payment, product-catalog, currency, or shipping.
2. If cart calls are slow, follow the cart runbook (cache connection pool exhausted).
3. If payment calls are slow or failing, follow the payment runbook (charge errors).
4. If every call is slow, check checkout CPU and scale out: `kubectl scale deploy/checkout --replicas=6`.
5. Escalate to team-checkout; if not resolved in 30 minutes, escalate to team-sre.

## Order events not published
Checkout publishes order events to Kafka. If publishing fails, orders still succeed but email, fraud-detection, and accounting stop receiving events. Check the Kafka runbook.
