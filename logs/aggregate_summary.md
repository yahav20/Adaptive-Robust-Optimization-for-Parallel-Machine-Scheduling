# Aggregated Benchmark Summary

- Run folders: 1
- Random rounds solved by both methods: 10/10
- Gap tolerance: 0.0001

## Random-Round Results

- Mean gap (Minimax - MILO): 0.0002
- Mean gap percentage: 0.0143%
- MILO lower by more than tolerance: 1
- Approximately equal: 9
- Minimax below MILO by more than tolerance: 0
- Different selected decisions (excluding ties as marked by the benchmark): 0
- Mean regret of Minimax decision: -0.0000
- Mean solve time: Minimax 0.025 s; MILO 0.071 s

## Per-Run Results

| Run | Solved / rounds | Mean gap | Mean gap % | MILO better | Equal | Theory violations | Mean Minimax (s) | Mean MILO (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run_20261009_164044 | 10 / 10 | 0.0002 | 0.0143% | 1 | 9 | 0 | 0.025 | 0.071 |

## Reference Checks

- Passed: 1 / 1 checked reference instances

## Notes

Gap is calculated as Minimax minus MILO. A positive value means MILO found a lower makespan; a negative value beyond tolerance is flagged as a theory-check violation.
