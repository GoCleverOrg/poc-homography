# Ortho floor extraction PoC — access & credentials

Sidebar to [ortho_floor_extraction.md](./ortho_floor_extraction.md). Lists every external platform, account, and stored credential the PoC depends on, plus where to rotate.

The PoC stays narrow on purpose: **one external service (Replicate)**, **one secrets store (Bitwarden Secrets Manager)**, **one git remote (GitHub)**, and the local filesystem. Other credentials surfaced in the shell environment (AWS, Cachix, Grafana Cloud, Supabase, etc.) belong to unrelated PedwebOrg / GoClever services and are NOT used by this PoC — their canonical list lives in `~/workspace/PedwebOrg/wiki/`.

## Replicate (model inference)

- **Web**: <https://replicate.com/smartterminal>
- **Account name**: `smartterminal` (Google login under the smartterminal org).
- **Account-token rotation**: <https://replicate.com/account/api-tokens>.
- **Documentation in the PedwebOrg wiki**: [`research/ml-hosting/replicate.md`](https://github.com/PedwebOrg/wiki/blob/main/research/ml-hosting/replicate.md) — canonical.

### Models used by this PoC

| model | role in pipeline | API ref |
|---|---|---|
| `prunaai/p-image-upscale` | upscale each 500×500 tile to ~2896×2896 (target 8 MPx) | https://replicate.com/prunaai/p-image-upscale |
| `chenxwh/depth-anything-v2` (Large) | inverse-depth per tile — first car-mask source | https://replicate.com/chenxwh/depth-anything-v2 |
| `pablodawson/segment-anything-automatic` (SAM v1 auto-mask) pinned to version `14fbb04535964b3d0c7fad03bb4ed272130f15b956cbedb7b2f20b5b8a2dbaa0` | class-agnostic region proposals — actual car-mask source | https://replicate.com/pablodawson/segment-anything-automatic |
| `david20321/depth-anything-v3-metric-large` pinned to version `e3523ab17a5e6f0e279933a6afdde67efe130bb9e7753cafc52a4b082257f46b` | tested as a depth alternative (T4); did NOT replace depth-anything-v2 in the production path | https://replicate.com/david20321/depth-anything-v3-metric-large |

### Approximate one-time cost

42 tiles × (1 upscale + 1 depth + 1 SAM) ≈ $0.40 for the entire site.

### Token storage

The API token is stored ONLY in Bitwarden Secrets Manager (next section). It must NEVER appear in source files, env files, commit messages, or chat transcripts. If a value is ever exposed (e.g. via `declare -x` env dump), rotate immediately at the URL above and then update bws via `bws secret edit <id> --value '<new>'`.

## Bitwarden Secrets Manager (`bws`)

- **Web**: <https://vault.bitwarden.com/#/sm>
- **CLI**: `bws` (already installed on dev machines via Nix; reference [`research/secrets/bitwarden.md`](https://github.com/PedwebOrg/wiki/blob/main/research/secrets/bitwarden.md) in the wiki for the full guide).
- **Machine-account access token**: `BWS_ACCESS_TOKEN` (per-machine; rotate at the web vault → Machine accounts → Access tokens).
- **Scoping rule**: per the wiki's two-level abstraction (org ↔ SM project ↔ secret prefix), this PoC's secret lives in the `GoClever` SM project as an un-prefixed *default* token.

### Project + secret used by this PoC

| field | value |
|---|---|
| bws organization id | `f6b9dfbc-6121-4d1f-84b2-b3af00b2e4b4` |
| bws project name | `GoClever` |
| bws project id | `19f59ce7-1fec-4885-b86b-b3b701408989` |
| secret key | `REPLICATE_API_DEFAULT_TOKEN` |
| secret id | `bc2602e3-ae89-478c-88df-b45a000bfde2` |
| consumer | every Replicate call in this PoC |

### Safe invocation shape

```sh
bws run --project-id 19f59ce7-1fec-4885-b86b-b3b701408989 -- .venv/bin/inv <task>
```

- `bws run` injects every secret in the project as a child-process env var.
- The Python entrypoints (`upscale.py`, `depth.py`, `sam_automask.py`, `depth_v3_metric.py`) read `REPLICATE_API_DEFAULT_TOKEN` and re-export it as `REPLICATE_API_TOKEN` in-process so the `replicate` client picks it up.
- **DO NOT** wrap with `bash -c '... $REPLICATE_API_DEFAULT_TOKEN ...'` or `sh -c '...'`. Bash will dump `declare -x` on error and leak the entire env. Lesson recorded in [ortho_floor_extraction.md § 7 Process lessons](./ortho_floor_extraction.md#7-process-lessons-recorded-for-next-iteration).
- **DO NOT** echo or log expansions of the token at any positional. The harness has been observed to silently drop stdout when it detects a token-shaped literal — output looks empty but the process still ran.

## GitHub (source control)

- **Mira repo**: <https://github.com/GoCleverOrg/mira>
- **Branch**: `poc/ortho-lines-534` (not pushed; lives only in the local worktree).
- **Worktree path**: `/Users/vasco/workspace/goclever/mira/.worktrees/poc-ortho-lines-534/`.
- **Access**: the standard `gh` CLI auth (used to push if/when the PoC is promoted).
- **No GitHub Actions / CI use this PoC**; it's a developer-local experiment.

## Local-only dependencies (no remote access)

| layer | how it's pinned |
|---|---|
| Python | `.venv/` in the PoC dir; `python3` from the Nix dev shell |
| Python deps | `requirements.txt` — `opencv-contrib-python` (NOT `opencv-python`, needed for `cv2.ximgproc`), `numpy`, `scikit-image`, `scipy`, `invoke`, `replicate`, `httpx`, `pyyaml` |
| Source map | `~/workspace/goclever/poc-homography/data/maps/icozee-cropped.tif` (this repo, DVC-tracked) |
| Generated artefacts | `/Users/vasco/workspace/goclever/mira/.worktrees/poc-ortho-lines-534/poc/ortho_lines_534/{map-tiles-*,out/*}` — all are `.gitignore`'d (regenerable end-to-end) |

## Explicitly NOT used by this PoC

The shell environment on the dev machine contains credentials for many other PedwebOrg / GoClever services (AWS, Cachix, Cloudflare, Supabase, Grafana Cloud, Gemini, Expo, Ansible Vault, etc.). **None are referenced by any script in this PoC.** Their canonical list and rotation paths live in `~/workspace/PedwebOrg/wiki/` (`research/secrets/bitwarden.md`, `tools/cloudflare-token-automation.md`, `tools/areapif-grafana-cloud.md`, `tools/account-master-credentials.md`).

## Rotation summary

| credential | rotate via | then | impact |
|---|---|---|---|
| `REPLICATE_API_DEFAULT_TOKEN` value | <https://replicate.com/account/api-tokens> | `bws secret edit bc2602e3-ae89-478c-88df-b45a000bfde2 --value '<new>'` | the next `bws run --project-id <id> -- inv <task>` picks it up automatically; no code change |
| `BWS_ACCESS_TOKEN` (your local machine-account access token) | <https://vault.bitwarden.com/#/sm> → Machine accounts → Access tokens | export the new value (Keychain on macOS) | the dev shell can read secrets again |
| GitHub token | `gh auth refresh` or rotate the PAT in GitHub settings | no PoC-side change | only matters if you push the branch |

## Out-of-band: how to grant a new contributor access

1. Add them to the `smartterminal` Replicate account (or, preferred, mint them a per-user token under the same account).
2. Add their machine account to the `GoClever` bws project read-permission (web vault → Machine accounts → assign project).
3. Give them this repo's read access (already true if they're in `GoCleverOrg`).
4. Point them at this doc and at [ortho_floor_extraction.md](./ortho_floor_extraction.md) for the pipeline reproduction recipe.
