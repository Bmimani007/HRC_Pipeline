# Cloud Deployment Guide — Sharing the Dashboard

This guide walks you through hosting the dashboard on **Streamlit Community Cloud** so your manager can access it from any browser via a URL, with a password gate protecting it.

**Total setup time:** about 15 minutes, one-time. After that, every time you push updated data to GitHub, the cloud dashboard updates automatically within ~30 seconds.

---

## What you'll end up with

```
                 your laptop
                 (edit data,
                  push update)
                       │
                       ▼
           ┌──────────────────────┐
           │  GitHub (private)    │
           │  free, your repo     │
           └──────────┬───────────┘
                      │ auto-pulls
                      ▼
          ┌────────────────────────┐
          │  Streamlit Cloud       │
          │  serves dashboard at:  │
          │  hrc-tata.streamlit.   │
          │  app                   │
          └──────────┬─────────────┘
                     │ HTTPS + password
                     ▼
              your manager
            (any laptop, phone)
```

---

## Step 1 — Create a GitHub account (skip if you already have one)

1. Go to **https://github.com**
2. Click **Sign up**, use your work email
3. Verify the email, finish onboarding (you can skip the survey)

You'll land on a dashboard. Done.

---

## Step 2 — Create a private repo and upload the project

There are two ways. The "no-terminal" way is easier for a non-coder.

### Option A — No terminal (recommended for you)

1. On GitHub, click the green **+** in the top-right → **New repository**
2. Set **Repository name**: `hrc-pipeline` (or any name)
3. Set visibility to **Private** (very important — your data is sensitive)
4. Tick **"Add a README file"** (just so the repo isn't empty)
5. Click **Create repository**
6. On the new repo page, click **Add file** → **Upload files**
7. Drag the **entire contents** of your `HRC_Pipeline` folder into the upload area
   - Important: drag the *contents*, not the parent folder itself. So `config.yaml`, `run.py`, the `data/`, `pipeline/`, `dashboard/`, `models/`, `report/` folders, etc. should all be at the top level of the repo.
8. Scroll down, type a commit message like `"Initial upload"`, click **Commit changes**

Wait for the upload to finish. You should see your project files listed in the repo.

**Important — verify the secrets file is NOT there.** Open the `.streamlit` folder in the repo. You should see only `config.toml` and `secrets.toml.example`. If you see `secrets.toml` (without `.example`), delete it from GitHub immediately — it would expose your password. (The `.gitignore` file should prevent this, but always double-check.)

### Option B — Terminal (skip if Option A worked)

If you're comfortable with Terminal, this is faster for ongoing updates. Skip this section unless you want to learn it.

```bash
cd ~/Documents/TATA\ STEEL/TS/dashboard/HRC_Pipeline
git init
git add .
git commit -m "Initial upload"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hrc-pipeline.git
git push -u origin main
```

---

## Step 3 — Sign up for Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **Continue with GitHub**, sign in with your GitHub account
3. Authorise Streamlit to access your repos when prompted
4. You'll land on the Streamlit Cloud dashboard

---

## Step 4 — Deploy your app

1. Click the **Create app** button (top-right)
2. Fill in:
   - **Repository**: select `YOUR_USERNAME/hrc-pipeline`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`  ← exactly this
   - **App URL** (optional): pick a custom subdomain like `hrc-tata` so the URL is `https://hrc-tata.streamlit.app` (otherwise you get a random one)
3. Click **Advanced settings**:
   - **Python version**: select **3.11** (most stable for the dependencies)
4. Click **Deploy**

Streamlit will now build the app — this takes 3–5 minutes the first time. It installs all the packages from `requirements.txt`, then starts the app. You'll see a streaming log of the build.

When build finishes, the page will refresh and show the **password gate**. The dashboard is live but locked.

---

## Step 5 — Set the password

The dashboard is now hosted but locked because you haven't configured a password yet. The login page is showing, but no password will work because nothing is set in the cloud.

1. On the Streamlit Cloud dashboard, click your app
2. Click the **⋮** menu (three dots, top-right) → **Settings**
3. In the left sidebar, click **Secrets**
4. In the text box, paste exactly:

   ```toml
   app_password = "your-strong-password-here"
   ```

   Replace `your-strong-password-here` with whatever you want. Make it strong — minimum 12 characters, mix of letters/numbers/symbols. Don't reuse a password you use elsewhere.
5. Click **Save**

The app will restart automatically (~20 seconds). Refresh the URL and try logging in with your password.

---

## Step 6 — Send the URL to your manager

Send your manager:

- **The URL** (e.g., `https://hrc-tata.streamlit.app`)
- **The password** (separately — e.g., via WhatsApp/Signal, not the same email)

That's it. Your manager opens the URL in any browser on any device, types the password, sees the dashboard.

---

## Updating the data later

Three steps each month after you receive new data:

1. **Edit `data/Raw_data.xlsx`** as before (add new month)
2. **Push the update to GitHub**:
   - On GitHub, navigate to the file → click the pencil icon → upload new version
   - Or via Terminal: `git add data/Raw_data.xlsx && git commit -m "Apr 2026 data" && git push`
3. **Streamlit Cloud auto-redeploys** within ~30–60 seconds. Manager just refreshes their browser.

The dashboard updates automatically — no need to message anyone.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build failed in Streamlit Cloud | Click "Manage app" → "Logs". Most likely a typo in `requirements.txt` or a missing file. |
| "Module not found: pipeline" | The `Main file path` in app settings must be `dashboard/app.py`, not just `app.py` |
| Password gate not appearing | Check the Secrets panel — must be exactly `app_password = "..."` with quotes |
| Dashboard works locally but fails on cloud | Cloud uses different Python version. In app settings → Advanced, set Python to **3.11** |
| Manager says "site not loading" | Streamlit Cloud free tier puts apps to sleep after 7 days of inactivity. First load takes ~30s to wake. |
| You want to change the password later | Settings → Secrets → edit `app_password` value, save. App restarts automatically. |

---

## Security notes

- **GitHub repo must be PRIVATE.** Public repos are visible to the world. If you set it to private in Step 2, you're fine.
- **The password is stored encrypted** in Streamlit Cloud's secrets store. It's not in your code.
- **Anyone with the URL + password can access the dashboard.** If your manager forwards it accidentally, anyone can open it. Treat the password like a real password.
- **To revoke access entirely**, just change the password in the Secrets panel. Anyone with the old password is locked out immediately.

---

## Cost

Streamlit Community Cloud is **free** for individual use:
- Unlimited public apps
- 1 GB RAM, 1 CPU per app — comfortable for this dashboard
- Apps sleep after 7 days inactivity (wake on first visit)

If you ever outgrow free tier, Streamlit's paid tier starts at ~$250/month — you won't need it for this use case.
