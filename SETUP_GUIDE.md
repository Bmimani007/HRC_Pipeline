# Setup Guide (Mac) — One-Time Setup

You'll do this **once**. After that, you only need two commands forever.

Total time: about 10 minutes.

---

## Step 1 — Install Python (skip if you already have it)

1. Go to **https://www.python.org/downloads/macos/**
2. Click the big yellow **"Download Python 3.12.x"** button (any 3.10+ works)
3. Open the downloaded `.pkg` file and follow the installer (just click Continue)
4. Restart your laptop after install

**To check it worked:** open Terminal (Cmd+Space, type "Terminal", press Enter) and type:
```bash
python3 --version
```
You should see something like `Python 3.12.1`. If you do, you're good.

---

## Step 2 — Get the project folder onto your laptop

You should have received a folder called `HRC_Pipeline`.
Move it somewhere easy to find — I suggest your Desktop or Documents.

---

## Step 3 — Open Terminal in the project folder

The fastest way:

1. Open **Finder**
2. Navigate to the `HRC_Pipeline` folder (don't open it, just select it)
3. Right-click on the folder → **"New Terminal at Folder"**

(If you don't see that option, go to System Settings → Keyboard → Keyboard Shortcuts → Services → Files and Folders → enable "New Terminal at Folder".)

Alternatively, in Terminal type `cd ` (with a space), then drag the folder onto the Terminal window, then press Enter.

To confirm you're in the right place, type:
```bash
ls
```
You should see: `config.yaml`, `data`, `dashboard`, `models`, `pipeline`, etc.

---

## Step 4 — Install the required packages

In the same Terminal, run:

```bash
pip3 install -r requirements.txt
```

This downloads about ~200 MB of analytical packages. It'll take 2–5 minutes depending on internet speed. Lots of text will scroll by — that's normal.

When it finishes (you get your prompt back, no red error text), you're done with setup forever.

---

## Step 5 — Run it for the first time

Still in Terminal, in the project folder:

```bash
python3 run.py
```

It'll print progress as it runs each region and each model. Takes 1–3 minutes. When done, open the report:

```bash
open outputs/HRC_Steel_Report.html
```

That's your report. Same one will refresh every time you re-run.

---

## Step 6 — Try the live dashboard

```bash
streamlit run dashboard/app.py
```

A browser tab opens automatically. Use the sidebar to switch regions, dates, drivers.

To stop the dashboard, go back to Terminal and press **Ctrl+C**.

---

## Forever after this

You only need two commands. Memorise (or stick on a Post-it):

| Want this                          | Run this                              |
|------------------------------------|---------------------------------------|
| Refresh the static HTML report     | `python3 run.py`                      |
| Open the live dashboard            | `streamlit run dashboard/app.py`      |

Both are run from inside the `HRC_Pipeline` folder in Terminal.

---

## Workflow once it's set up

```
   ┌──────────────────────────────┐
   │ 1. Open data/Raw_data.xlsx   │   (in Excel as normal)
   │    Add new month / variable  │
   │    Save & close              │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 2. Open Terminal in folder   │
   │    Run: python3 run.py       │   (~2 minutes)
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 3. Open the HTML report      │
   │    Or: streamlit run ...     │   (for live view)
   └──────────────────────────────┘
```

---

## If something goes wrong

**"command not found: python3"**
Python isn't installed or wasn't restarted. Redo Step 1 and restart your laptop.

**"command not found: pip3"**
Same — Python install didn't complete. Reinstall.

**"command not found: streamlit"** (after running `pip3 install`)
The packages didn't install. Try:
```bash
pip3 install --user -r requirements.txt
```

**Red error wall when running `python3 run.py`**
The first line of the error usually tells you what file or column is missing. Check `config.yaml` matches the actual column names in the xlsx exactly. Spaces and capitalisation matter.

**Anything else**
Take a screenshot of the Terminal window and send it back — the answer is almost always one line.
