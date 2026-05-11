# HRC Pipeline — Liquidity Module Build Notes

**Build date:** 11 May 2026
**Build scope:** India Liquidity Monitor (Section 1-7 of original spec, scope “Core + stress gauge + regime performance table”)

---

## What changed

### Data
* **`data/Raw_data.xlsx` → India sheet** now has 9 columns × 88 rows (was 4 × 87). New columns:
  * `WACR` — official RBI Weighted Average Call Money Rate (monthly average)
  * `Repo_Rate` — Policy Repo Rate (%, month-end)
  * `GSec_10Y` — 10-Year G-Sec yield (%, monthly average)
  * `CRR` — Cash Reserve Ratio (%, month-end)
  * `Bank_Credit` — Bank Credit outstanding (₹ Crore, fortnightly last)
* Historical data Jan 2019 → Apr 2026. Apr 2026 row has liquidity but no HRC; the pipeline drops it automatically until HRC is added.

### Code
* **`pipeline/liquidity.py`** — NEW module. Contains: derived-series computation (WACR_Spread, Bank_Credit_YoY, etc.), regime classification (Tight/Neutral/Surplus), 0-100 stress index, RBI policy regime detection, performance-by-regime stats, lead-lag analysis, current-state summarizer, interpretation engine.
* **`pipeline/data_loader.py`** — Recognizes a new `liquidity:` block per region. Auto-computes derived series on load. Excludes raw liquidity columns from the auto-driver list, keeping VIF clean. Adds `WACR_Spread` as a regular driver alongside the existing 2 (so 3 India drivers total).
* **`pipeline/lead_lag.py`** & **`pipeline/diagnostics.py`** — Bug fix for empty-driver case (was previously crashing the orchestrator on the US region; the failure was latent but harmless). Now returns empty DataFrames cleanly.
* **`config.yaml`** — Added `india.liquidity` block listing the 5 base columns and the single driver column (WACR_Spread).
* **`dashboard/app.py`** — Conditional 11th tab “💧 Liquidity” shown only for regions with a `liquidity` config block (currently India only). China/US tab strips are unchanged.
* **`data/glossary.yaml`** — 14 new entries (WACR, WACR Spread, Repo Rate, MSF, SDF, G-Sec 10Y, CRR, Bank Credit YoY, Liquidity Regime, Policy Regime, Stress Index, Term Premium, VRR, Banking System Liquidity).
* **`README.md`** — Monthly RBI workflow section added.

---

## What the new tab shows (India only)

1. **Macro state header** — 4 metric cards (regime / spread / stress / policy) + an interpretation paragraph generated from current readings.
2. **Stress gauge** — Plotly 0-100 indicator with 5 colored bands.
3. **HRC vs Liquidity Regime chart** — HRC line overlaid with shaded regime backgrounds (green/grey/red) and a dashed WACR-Repo spread line.
4. **Regime performance table** — annualised return, volatility, drawdown, hit rate per regime, colour-graded by return.
5. **Lead-lag heatmap** — Pearson correlations of HRC returns vs lagged liquidity variables at 0/1/3/6/12 months.
6. **Liquidity variable panel** — 6 mini-charts: WACR, Repo, 10Y G-Sec, CRR, WACR Spread, Bank Credit YoY.

Each section has an `📝 Interpretation` expander with conditional prose.

---

## Empirical validation (real findings from the build)

The data tells a clean macro story:

| Regime  | Months | Ann. Return | Volatility | Max DD | Hit Rate |
|---------|--------|------------:|-----------:|-------:|---------:|
| Surplus | 26     | **+32.8%**  | 20.9%      | -8.2%  | 61.5%    |
| Neutral | 22     | -7.5%       | 11.7%      | -20.3% | 36.4%    |
| Tight   | 38     | -3.7%       | 12.3%      | -26.4% | 42.1%    |

A 36 percentage-point spread between surplus and tight regimes. Hit rates back it up (62% vs 42%).

VIF for the new driver set is 1.30 / 1.24 / 1.22 — clean, no collinearity. WACR_Spread correlates -0.25 with HRC contemporaneously (right sign, right magnitude). The auto-detected regime blocks match macro reality: COVID-era 24-month surplus (May 2020 – Apr 2022), RBI tightening 30-month tight stretch (Oct 2022 – Mar 2025), 2025 onward oscillating around neutral as RBI eases.

---

## Monthly workflow (going forward)

Same as before — edit one Excel file, run one command — **plus 5 RBI cells per month**.

1. Download from RBI DBIE:
   * `Weighted_Average_Call_Money_Rates.xlsx`
   * `50_Macroeconomic_Indicators.xlsx`
2. In `data/Raw_data.xlsx` India sheet, add the new month's row with the 5 liquidity values (in addition to your usual HRC + iron ore + HCC values). The README has the exact mapping.
3. `git push` to redeploy. Streamlit Cloud auto-rebuilds in ~30 seconds.

Derived series (WACR_Spread, Stress_Index, regime labels, etc.) are computed automatically by the pipeline — you don't enter them.

---

## Deployment

**Critical (per project memory):** replace the **entire** `HRC_Pipeline/` folder. Partial uploads cause the "mismatched state" failure mode (e.g., new `config.yaml` referencing a column that the older `data_loader.py` doesn't know how to compute).

Steps:
1. Back up your current `HRC_Pipeline/` folder (just in case).
2. Replace it entirely with the unzipped contents of `HRC_Pipeline_with_liquidity.zip`.
3. From terminal:
   ```bash
   cd ~/Documents/TATA\ STEEL/TS/dashboard/HRC_Pipeline
   git add .
   git commit -m "Add India Liquidity Monitor (WACR/Repo/G-Sec data + new tab)"
   git push
   ```
4. Streamlit Cloud will auto-rebuild. Wait ~30s, then visit `hrcpipeline-tata-bharat.streamlit.app` and click India in the sidebar — you should see the new 💧 Liquidity tab.

If the Streamlit deployment fails: check the Streamlit Cloud logs. Most likely cause is a missing dependency, but the requirements.txt is unchanged (no new packages — `liquidity.py` uses only pandas/numpy already in requirements).

---

## What was NOT built (deferred from spec)

* Stress gauge in absolute headline-card form (built as a Plotly gauge instead — same idea, integrated into tab)
* Macro cards section (replaced by the 4-metric header — leaner equivalent)
* RBI policy timeline visualization (out of scope as discussed; the HRC vs Liquidity chart shows the same information via shaded backgrounds)
* Lead-lag heatmap (built as a colour-graded table instead — same information, more readable)
* VRR/VRRR operations data (not available in your RBI files; mentioned in glossary)
* Banking system liquidity surplus/deficit (₹ crore) (not available in your files; the WACR spread captures the same information indirectly)
* PMI (not in RBI files — it's S&P Global)

These can be added later if needed; the architecture supports them. Adding new liquidity variables means appending a column to the India sheet, adding it to `india.liquidity.base_columns` in config.yaml, and (optionally) adding it to `driver_columns`.
