# Cleanup for handoff

Preview generated cache cleanup:

```powershell
python tools\cleanup_handoff.py
```

Remove only the listed generated caches:

```powershell
python tools\cleanup_handoff.py --execute
```

The cleanup script is intentionally narrow. It does not remove datasets,
models, uploads, SQLite databases or virtual environments.

Source handoff should include:

```text
app.py
config.py
requirements.txt
.env.example
README.md
mddad/
templates/
static/
tests/
tools/verify_driver_safety_v1.py
tools/cleanup_handoff.py
docs/HANDOFF_FOR_NEW_USER.md
docs/PROJECT_CLEANUP_AUDIT.md
docs/PRODUCT_ROADMAP.md
models/driver_safety_v1/manifest.json
models/driver_safety_v1/README.md
```

Ship the four binary artifacts in `models/driver_safety_v1/` separately. The
large research datasets, MMRT/Activity7 bundles, generated reports and unrelated
`tools/car_logo_dataset/` experiment are not required to run the product path.
