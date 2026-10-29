# Benchmarking and autotuning

Benchmarks are explicitly invoked, separate from request execution:

```bash
oflow benchmark --items 128 --warmups 1 --repeats 3 --output reports/local-benchmark.json
```

Seven portable scenarios cover one interactive client, multiple interactive clients, bulk work, mixed work, backend failures, deadline expiration and repeated requests/memory. Scenario order is shuffled for each repetition using a recorded seed. Warmups are excluded from measured rows. Reports record environment, runtime/backend/model identifier, workload/configuration, status counts, correctness, throughput, p50/p95/p99 latency, queue p95 and tracemalloc peak/retained Python bytes.

Tracemalloc is not RSS, native allocator memory, pinned-memory accounting or VRAM. This short synthetic workload is not a soak test and cannot prove long-run memory stability. Request throughput includes errors/timeouts; completed_per_second separately counts successes. No token-throughput metric is fabricated for workloads without token semantics.

The committed [CPU benchmark](reports/cpu-scheduler-windows.json) measured Windows 11, Python 3.12.2, Pydantic 2.14.1, 16 logical CPU threads, approximately 31.92 GiB RAM, deterministic mock backend, no trained model/GPU. Each scenario has 128 requests × 3 measured repetitions after one warmup. All correctness gates passed. Windows timer granularity, instrumentation and mock delays materially affect these numbers; they are not portable model-inference performance claims.

| Scenario | Median requests/s | Median per-run p95 latency |
|---|---:|---:|
| Single interactive | 60.8 | 16 ms |
| Multiple interactive | 118.0 | 94 ms |
| Bulk | 113.1 | 203 ms |
| Mixed | 122.5 | 79 ms |
| Failures | 127.1 | 63 ms |
| Deadlines | 268.4 | 16 ms |
| Repeated memory | 125.1 | 63 ms |

Peak traced Python allocations were approximately 323–579 KiB per run. Deadline polling can overshoot a 1 ms deadline on this host; the report records terminal timeout handling, not 1 ms real-time compliance.

The generic autotune helper requires at least one warmup, three measured repetitions and explicit environment metadata. It shuffles candidates, compares every warmup/measured output against the reference, and excludes a candidate if any comparison or execution fails. No equivalent candidate means winner=null. Full structure, labels and exact integers/booleans are checked; finite floats use absolute plus relative tolerance. It does not automatically apply results to a different environment.

For an actual model, include model/checkpoint revision, tokenizer, hardware UUID/driver/runtime, precision, shapes, memory, workload and configuration in environment metadata; synchronize accelerator timing within candidate functions. Use independent model/reference fixtures where available. Default batch sizes and efficiency thresholds are starting points, not universal optima.
