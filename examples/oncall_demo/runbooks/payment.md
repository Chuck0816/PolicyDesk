---
doc_type: runbook
service: payment
title: Payment runbook
---
# Payment runbook

## Charge errors
Alert: PaymentErrorRateHigh. Charge error ratio above 2%.

1. Check whether errors are from the card gateway (HTTP 5xx from the gateway) or from payment itself (validation errors).
2. If a config or feature flag changed in the last hour, roll it back first.
3. If the gateway is failing, enable the fallback gateway flag `payment.fallback_gateway=true`.
4. Owner: team-payments, escalation team-sre. Checkout will show errors while payment fails.

## Slow charges
Charges above 2 s usually come from gateway latency. Do not restart payment; it does not help.
