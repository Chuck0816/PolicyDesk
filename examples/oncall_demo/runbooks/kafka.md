---
doc_type: runbook
service: kafka
title: Kafka runbook
---
# Kafka runbook

## Consumer lag
Alert: KafkaConsumerLagHigh. A consumer group lags more than 10000 messages.

1. Find the lagging consumer group: fraud-detection, email, or accounting.
2. Check whether the consumer is crashing (OOM, restarts) or just slow.
3. Scale the consumer: `kubectl scale deploy/fraud-detection --replicas=4`.
4. Kafka itself is owned by team-sre; the consumers are owned by their teams.

## Broker disk full
If brokers are above 85% disk, reduce retention on the orders topic to 24 hours.
