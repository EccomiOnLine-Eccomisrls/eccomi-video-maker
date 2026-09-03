# EVS CPU Compositor — Render Workflows

Render Workflow used by EVS PRO for NON-GPU corrections such as voice, music, text/timing and light compositing.

## Render Dashboard configuration

- Service type: Workflow
- Repository: `EccomiOnLine-Eccomisrls/eccomi-video-maker`
- Branch: `main`
- Root Directory: `cpu_workflow`
- Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `python main.py`
- Task: `remix_master`
- Compute plan: `flex` (declared in code)

No GPU is required. The task stream-copies the existing video track and rebuilds only the audio mix.

## Triggering

Supabase `evs-correction-run` triggers this task through the Render Workflows API/SDK. The task receives only URLs, a short-lived Supabase signed upload token, the public Supabase anon key, and correction parameters.

The task uploads the new MP4 to the existing public `videos` bucket and calls the existing `evs-video-callback`, returning the corrected master to the standard EVS QA and Release Gate flow.

## Security

- No Supabase service-role key is stored on Render.
- Signed upload tokens are short-lived.
- The Supabase anon key is public by design and is used only to satisfy the JWT-protected callback gateway.
- Customer delivery remains blocked until the EVS Release Gate is approved.
