# Lab 2 - Prompt and Pipeline Grid

## Run and scope

Refreshed with `python labs/lab2/grid.py --all --split dev --save reports/lab2_grid.json` on 2026-10-05. Seven variants were evaluated on the 60-example development split. Values below are from the saved JSON; costs and p95 latency are measured only where calls were uncached.

## Results

| Variant | Field accuracy | Record accuracy | Schema valid | Cost (USD) | p95 (ms) | Cached calls |
|---|---:|---:|---:|---:|---:|---:|
| zero_shot | 0.919 | 0.483 | 1.000 | 0.0457 | 1,886 | 0/60 |
| zero_shot_main | 0.917 | 0.533 | 1.000 | 0.2239 | 6,007 | 0/60 |
| few_shot | 0.925 | 0.550 | 1.000 | 0.0620 | 1,948 | 0/60 |
| few_shot_main | 0.944 | 0.667 | 1.000 | 0.2581 | 5,781 | 0/60 |
| few_shot_reasoned | 0.917 | 0.483 | 1.000 | 0.0784 | 2,144 | 0/60 |
| few_shot_reasoned_main | 0.940 | 0.667 | 1.000 | 0.2550 | 5,362 | 0/60 |
| cascade | 0.925 | 0.550 | 1.000 | 0.0000 | 0 | 60/60 |

All variants had zero recorded errors and 1.000 schema validity. The cascade made no escalations in this run. All 60 cascade outputs were cache hits, so its zero cost and latency do not establish live cascade economics or escalation behavior.

## Paired comparisons

Against zero_shot, the saved paired tests found no significant change for zero_shot_main (p=0.6476), few_shot (p=0.3438), few_shot_reasoned (p=1.0000), or cascade (p=0.3438). few_shot_main improved record accuracy significantly (p=0.0266), as did few_shot_reasoned_main (p=0.0347). These comparisons use the same 60 development cases.

## Interpretation

few_shot_main reached the top record accuracy, tied with few_shot_reasoned_main at 0.667, but cost about 4.2 times as much as few_shot and had roughly three times its p95 latency. The paired result supports a quality gain for the main-model variants, with a material cost and latency trade-off. few_shot is a lower-cost choice when 0.55 record accuracy is sufficient. Do not select the cascade from this cached run: it escalated 0/60 times.

## Artifact

Raw grid, per-case output, budgets, and paired-test results: `reports/lab2_grid.json`.
