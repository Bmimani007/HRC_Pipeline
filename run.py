#!/usr/bin/env python3
"""
HRC Steel Pipeline — main entry point.

Usage:
    python3 run.py # uses config.yaml in current dir
    python3 run.py path/to/config.yaml

What it does:
    1. Reads config.yaml
    2. Loads data from xlsx (auto-detects columns)
    3. Runs all enabled analyses + models on every enabled region
    4. Saves results.json (consumed by the dashboard)
    5. Generates the HTML report

After it finishes:
    • Open outputs/HRC_Steel_Report.html in any browser
    • Or run: streamlit run dashboard/app.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import yaml

# Make sure the project root is on the import path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import run_pipeline, save_results_json
from report.builder import build_report


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not Path(config_path).exists():
        print(f"(fail) Config file not found: {config_path}")
        print(f" Make sure you're running this from the project folder.")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ---- Run the pipeline ----
    import traceback
    results = None
    try:
        results = run_pipeline(config)
    except Exception as e:
        print(f"\n(fail) Pipeline failed: {type(e).__name__}: {e}")
        print("\n--- Full traceback ---")
        traceback.print_exc()
        print("\nCheck that:")
        print(f" • The data file exists at: {config['data']['file']}")
        print(f" • The sheets and columns in the file match config.yaml")
        print(f" • All packages are installed: pip3 install -r requirements.txt")
        sys.exit(1)

    # ---- Save JSON ---- (best-effort; never block the report on this)
    print(f"\n[Saving outputs]")
    try:
        json_path = config["output"]["results_json"]
        save_results_json(results, json_path)
    except Exception as e:
        print(f" (!) Failed to save JSON: {type(e).__name__}: {e}")
        print(f" (continuing — JSON is only used by the dashboard)")

    # ---- Build HTML report ---- (this MUST succeed; it's the user's deliverable)
    print(f" • Building HTML report...", end="", flush=True)
    t0 = time.time()
    report_path = config["output"]["report_path"]
    try:
        report_path = build_report(results, report_path)
        print(f" (ok) ({time.time() - t0:.1f}s)")
        print(f" (ok) Report saved: {report_path}")
    except Exception as e:
        print(f" ✗ FAILED")
        # Write a minimal fallback HTML so something exists at the expected path.
        # This way `open outputs/HRC_Steel_Report.html` doesn't fail with "file not found".
        try:
            tb = traceback.format_exc()
            fallback = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>HRC Report — Build Failed</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
       margin: 40px auto; padding: 24px; color: #1A1F2E; }}
h1 {{ color: #A4161A; }}
pre {{ background: #F8FAFC; padding: 16px; border-radius: 6px; overflow-x: auto;
      border: 1px solid #E5E9F0; font-size: 12px; line-height: 1.5; }}
.tip {{ background: #FFF8E1; padding: 12px 16px; border-left: 3px solid #C9540F;
       border-radius: 4px; margin: 16px 0; }}
</style></head><body>
<h1>Report build failed</h1>
<p>The pipeline ran successfully but the HTML report builder crashed. The data
   has been preserved in <code>outputs/results.json</code>; only the visual
   report failed to render.</p>
<div class="tip">Send the traceback below back to your assistant — it'll usually
  be a one-line fix.</div>
<h2>Error</h2>
<p><b>{type(e).__name__}:</b> {str(e)}</p>
<h2>Traceback</h2>
<pre>{tb}</pre>
</body></html>"""
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                f.write(fallback)
            print(f" → Wrote fallback report to {report_path} so you can see the error.")
        except Exception:
            pass # If even the fallback can't write, give up gracefully
        print(f"\n--- Report builder traceback ---")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'─' * 70}")
    print(f"(ok) DONE")
    print(f"{'─' * 70}")
    print(f"\n Open the report: open {report_path}")
    print(f" Open the dashboard: streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    main()
