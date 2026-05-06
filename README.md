# HRC Steel Intelligence Pipeline

A self-running analytical pipeline for HRC steel prices and spreads.
Edit one Excel file, run one command, get an updated report and dashboard.

---

## What you get

Two ways to consume the analysis:

1. **HTML report** (`outputs/HRC_Steel_Report.html`) — polished, self-contained,
   shareable with leadership. Open in any browser.
2. **Live dashboard** (Streamlit) — interactive, lets you toggle drivers,
   change date ranges, switch regions on the fly.

Both are powered by the **same** pipeline reading the **same** Excel file.

Currently configured for two regions:
- **China** — HRC China FOB prices in USD, 9 drivers (raw materials + macro)
- **India** — HRC FBD prices in INR, with raw materials (Iron Ore Odisha + Import HCC India CFR)

---

## What you actually do (90% of the time)

Just three things, in order:

### 1. Edit `data/Raw_data.xlsx`

This is the only file with your data. Open it like any Excel file. It has three sheets:

| Sheet     | Purpose                                                  |
|-----------|----------------------------------------------------------|
| README    | Quick reference (how to update, formulas used)           |
| China     | Monthly prices in USD + macro indicators                 |
| India     | Monthly prices in INR                                    |

**Adding a new month:** Add a row at the bottom. Just type the date in column A and the values across.

**Adding a new variable:** Add a new column. Give it a clear name in row 1.
The pipeline auto-detects new columns and includes them as drivers — no code edit needed.

**Removing a variable:** Delete the column. The pipeline runs without it.

**Don't rename:** the `Month` column or the target columns (anything containing "HRC" in the header).

### 2. Refresh the report

Open Terminal, navigate to the project folder, and run:

```bash
python3 run.py
```

Takes 1–3 minutes. When it's done, open `outputs/HRC_Steel_Report.html` in your browser.

### 3. Open the live dashboard

For interactive exploration:

```bash
streamlit run dashboard/app.py
```

A browser tab opens automatically. Edit the Excel file, click 🔄 **Refresh data** in the sidebar, and the dashboard re-runs the analysis on the new data.

---

## What's in the report

| Section              | What it shows                                                   |
|----------------------|------------------------------------------------------------------|
| Cover                | High-level KPIs across all regions                              |
| Region Overview      | HRC price history per region                                    |
| **Spread Analysis**  | Monthly spread, FY averages, percentiles, decomposition         |
| Diagnostics          | Correlation matrix, multicollinearity (VIF), stationarity (ADF) |
| Lead/Lag             | CCF & Granger causality — which drivers lead price?             |
| Models               | ARIMAX + ARDL forecasts with 95% CI, GARCH volatility           |
| Regimes              | K-means classification of market states                         |
| Attribution          | Rolling regression — driver importance over time                |
| Events               | Pre/post analysis around key episodes (e.g., Ukraine war)       |
| **Cross-Region**     | China vs India spread comparison                                |

The spread formula matches your Tata BPM deck:
```
Spread = HRC − (1.6 × Iron Ore + 0.9 × HCC)
```

---

## When something doesn't work

The pipeline tries to give plain-English errors. The most common ones:

| Error contains…                          | What to do                                   |
|------------------------------------------|----------------------------------------------|
| "Data file not found"                    | Check the file is at `data/Raw_data.xlsx`   |
| "Target X not found"                     | Check the target column name in `config.yaml` matches the column header in the xlsx exactly (including spaces and capitalisation) |
| "Sheet 'X' not found"                    | Check the sheet name in `config.yaml` matches the actual tab in the xlsx |
| "Missing package"                        | Run `pip3 install -r requirements.txt`      |
| Anything about statsmodels / arch        | Same — install requirements                 |
| "Only N observations"                    | You need at least 24 months of data         |

---

## Customisation (when you're ready)

Open `config.yaml` in any text editor. Comments explain what each block does.

**To turn off a region:** set `enabled: false` under that region.

**To turn off a model:** set `enabled: false` under that model.

**To change which model uses which region:** edit the `regions` list under each model.

**To add a new event for analysis:** add an entry under `analysis.events.episodes`.

**To swap a model entirely:** drop a new file in `models/` that follows the
template in `models/base.py`, register it with `@register_model("your_name")`,
add a config block under `models:`. Done — no other file needs changes.

---

## Folder structure

```
HRC_Pipeline/
├── config.yaml              ← edit to change behaviour
├── requirements.txt         ← package list for setup
├── run.py                   ← THE entry point
│
├── data/
│   └── Raw_data.xlsx        ← edit to add data
│
├── pipeline/                ← analytical engine
│   ├── data_loader.py
│   ├── diagnostics.py
│   ├── lead_lag.py
│   ├── spread.py
│   ├── regimes.py
│   ├── attribution.py
│   ├── events.py
│   └── orchestrator.py
│
├── models/                  ← swappable forecasting models
│   ├── base.py              ← interface (don't edit unless adding new models)
│   ├── registry.py          ← auto-discovery (don't edit)
│   ├── arimax.py
│   ├── ardl.py
│   └── garch.py
│
├── report/
│   └── builder.py           ← HTML report generator
│
├── dashboard/
│   └── app.py               ← Streamlit dashboard
│
└── outputs/                 ← created automatically on each run
    ├── HRC_Steel_Report.html
    └── results.json
```

---

## First-time setup (Mac)

See `SETUP_GUIDE.md` — one page, step-by-step.
