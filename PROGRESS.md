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
- Added historical priors, complete candidate generation, smoothing, and safe
  fallback estimates for transitions missing from the history.
- Integrated `priors.build_candidates(...)` with the stable strategy contract.
- Integrated mock evaluation over 10 seeds: 10/10 positive runs, median net
  gain 2,956,653 and minimum net gain 2,411,101. These values validate the
  integration only; the hidden judging effects are intentionally different.
- Diversified the initial exploration: 12 history-backed pilots, one
  target-history fallback and one tariff-price fallback. Repeated pilots still
  dominate final decisions.
- Added deterministic prior tests, exact final budget/contact checks and an A/B
  benchmark against the tariff-fit fallback.
- Extended mock evaluation over 50 seeds: 50/50 positive runs, median net gain
  2,631,417 and minimum net gain 2,009,929. These values validate stability,
  not the hidden judging score.
- Added `TESTING.md` so the full test, benchmark and submission workflow can be
  reproduced manually.
- Next step: add presentation diagnostics without changing the judging logic.
