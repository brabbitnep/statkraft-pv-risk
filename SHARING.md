# 🚀 How to Deploy This App (Permanent)

You can deploy this app to **Streamlit Community Cloud** for free. This gives you a permanent link (e.g., `https://your-app.streamlit.app`) that stays online 24/7.

## Step 1: Push to GitHub
1.  **Create a new repository** on [GitHub](https://github.com/new).
    *   Name it something like `statkraft-pv-risk`.
    *   Select **Public** (or Private if you prefer).
    *   **Do NOT** initialize with README, .gitignore, or License (we already have them).
2.  **Push your code** by running these commands in your terminal (replace `YOUR_USERNAME` with your actual GitHub username):
    ```powershell
    git remote add origin https://github.com/YOUR_USERNAME/statkraft-pv-risk.git
    git branch -M main
    git push -u origin main
    ```

## Step 2: Deploy on Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Log in with GitHub.
3.  Click **"New App"**.
4.  Select your `statkraft-pv-risk` repository.
5.  Ensure the settings are:
    *   **Main file path:** `app.py`
6.  Click **Deploy**!

---

## Alternative: Local Testing (Temporary)
If you just want to share a quick link without deploying, use **Serveo**:
1.  Run in a new terminal: `ssh -R 80:localhost:8501 serveo.net`
