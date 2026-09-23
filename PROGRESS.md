# Progress log

## 2026-09-23

- Confirmed that the hackathon starter kit is available in the repository.
- Verified that local environment files and API credentials are excluded by `.gitignore`.
- Implemented the first adaptive campaign-agent baseline in `agent.py` and
  `strategy.py`.
- Added a stable contract for the independently developed `priors.py` module.
- Generated a reproducible `submission.csv` and added a smoke test for budget,
  contact, pilot and output limits.
- Local mock evaluation over 10 seeds: 10/10 positive runs, median net gain
  649,011 and minimum net gain 115,569.
- Next step: integrate hierarchical historical priors and compare the result
  against the tariff-fit fallback without tuning to the mock effects.
