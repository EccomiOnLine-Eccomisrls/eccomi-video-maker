# EVS CPU Compositor — Render Workflows

Render Workflow used by EVS PRO for NON-GPU production and corrections.

## Render Dashboard configuration

- Service type: Workflow
- Repository: `EccomiOnLine-Eccomisrls/eccomi-video-maker`
- Branch: `main`
- Root Directory: `cpu_workflow`
- Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python bootstrap.py`
- Compute plan: `flex` (declared in code)

`bootstrap.py` deliberately imports the historical `main.py` unchanged and then registers the additional commercial task. This keeps the validated correction compositor isolated from new production routes.

## Tasks

### `remix_master`
Historical validated task used for NON-GPU corrections and remixes from an existing master. Its implementation remains in `main.py`.

### `create_commercial`
Isolated NON-GPU initial-production task for PRODUCT / SERVICE orders without a mascot. It uses real customer assets, deterministic brand/logo/CTA composition, voice and music, then returns the master to the standard EVS QA and Release Gate flow.

Task identifiers:

- `evs-cpu-compositor/remix_master`
- `evs-cpu-compositor/create_commercial`

## Routing rule

- Mascot required → legacy GPU route (RunPod), unchanged.
- Product / Service without mascot → `create_commercial` on Render Flex.
- No automatic fallback is allowed between routes.

## Security

- No Supabase service-role key is stored on Render.
- Signed upload tokens are short-lived.
- The Supabase anon key is public by design and is used only to satisfy the JWT-protected callback gateway.
- Customer delivery remains blocked until the EVS Release Gate is approved.
