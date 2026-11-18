# Demo test questions

Import order: `service_catalog.yaml` + `alerts.yaml` → `runbooks/*.md` → `postmortems/*.md`.
Ask each question with **compare** on (hybrid vs hybrid + graph). "Graph helps" marks
questions where the answer needs multi-hop dependencies or facts spread over documents.

| # | Question | Expected answer | Graph helps |
| --- | --- | --- | --- |
| 1 | 如果 payment 不可用，会影响哪些服务？ | checkout (1 hop), frontend (2 hops via checkout) | Yes (multi-hop impact) |
| 2 | 商品目录变慢会影响哪些服务？ | frontend, recommendation, checkout (1 hop) | Yes |
| 3 | If postgresql has lock contention, which services are affected? | product-catalog, accounting (1 hop); frontend, recommendation, checkout (2 hops) | Yes (3-hop chain) |
| 4 | Kafka 出问题会影响谁？ | checkout (publishes), email, fraud-detection, accounting; frontend via checkout | Yes |
| 5 | 谁负责 valkey-cart？升级找谁？ | team-checkout → team-sre | Some (owner + escalation) |
| 6 | Who owns the currency service and what is the escalation path? | team-payments → team-sre | Some |
| 7 | checkout P99 延迟飙升，同时 payment 错误率上升，最可能的根因是什么？ | payment (shared dependency; INC-101, INC-105); check gateway / recent config, fallback gateway flag | Yes (root-cause ranking + history) |
| 8 | checkout 延迟升高，同时 cart 超时，根因可能是什么？ | cart / valkey-cart connection exhaustion (INC-102); restart cart pods, roll back release | Yes |
| 9 | frontend 页面很慢，recommendation 也超时，根因是什么？ | product-catalog → postgresql lock contention (INC-104); cancel blocking query | Yes (shared downstream) |
| 10 | 以前有没有发生过类似的风控服务消费积压？当时怎么解决的？ | INC-103: fraud-detection OOM; raised memory, scaled consumers 2→4 | Yes (similar incidents) |
| 11 | CheckoutLatencyHigh 告警了，我该怎么处理？ | checkout runbook § Latency high: find slow downstream, follow cart/payment runbook, scale out, escalate | Some (alert → runbook) |
| 12 | KafkaConsumerLagHigh is firing. What should I do? | Kafka runbook § Consumer lag: find group, check OOM, scale consumer | Some |
| 13 | valkey-cart 能直接重启吗？ | Not casually: no persistence, all carts lost; tell support first (cart runbook) | No (plain text lookup) |
| 14 | payment 的 SLO 是多少？ | 99.9% of charges succeed | No (catalog text) |
| 15 | 支付失败时可以重启 payment 吗？ | Slow charges come from the gateway; restarting does not help. Roll back config or enable fallback gateway | No |
| 16 | 怎么部署一个新的 Kubernetes 集群？ | Should say the documents do not cover it | Refusal check |

Things to look at for each answer: the graph reasoning paths shown under the sources, whether
the hidden structured result lists the expected `root_cause_services` / `affected_services` /
`owner_team`, and the timing line (graph modes add Neo4j queries but no extra LLM call).
