# 🚀 Deployment & Update Guide

## 1. The Update Workflow

When you make changes to your code (Frontend or Backend), follow this cycle:

### Step 1: Code Locally
Make your edits on your local machine. Test them using the development setup (`docker compose up`).

### Step 2: Transfer Code to Server
You need to get your new code onto the VPS.
*   **Method A (Recommended): Git**
    1.  Commit and push changes to GitHub/GitLab.
    2.  SSH into your server.
    3.  Run `git pull origin main` in your project folder.
*   **Method B: Manual Copy (SCP/SFTP)**
    1.  Copy the modified files/folders to the server using a tool like FileZilla or `scp`.

### Step 3: Rebuild & Restart
On your server, run this **single command** to apply changes:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

**Why this command?**
*   `--build`: Forces Docker to re-compile your application with the new code.
*   `up -d`: Starts the containers in the background. If containers are already running, it only recreates the ones that changed.

---

### "Something is stuck / clean start"
If you need to fully reset:
```bash
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d --build
```
