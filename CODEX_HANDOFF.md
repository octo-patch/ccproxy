# Codex Handoff: remaining follow-ups

## Current status

The packaged default-shape work is complete for the supported public defaults:

- `src/ccproxy/templates/shapes/anthropic.mflow`
- `src/ccproxy/templates/shapes/gemini.mflow`

Those artifacts were captured from real CLI traffic, repackaged through the shared shaping
machinery, audited as request-only `.mflow` files, and verified through `just e2e-packaged-mflows`.

No active blocker from the previous packaged-shape handoff remains.

## Remaining follow-ups only

- `src/ccproxy/transport/sidecar.py:_HOP_BY_HOP` is still a cosmetic misnomer because it includes
  `host` and `content-length`, which are not strictly RFC 7230 hop-by-hop headers.
- Codex/OpenAI Responses is not a packaged default.
  Do not add it back to `nix/defaults.nix`, `scripts/package_mflows.py`, or the packaged-shape E2E
  gate until ccproxy has live supported OpenAI Responses/Codex provider behavior.
