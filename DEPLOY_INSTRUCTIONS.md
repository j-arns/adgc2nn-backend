# 🌍 Production Server Deployment Guide

Use this guide when you have a fresh Virtual Private Server (VPS) and want to deploy the `adGC2NN` Full Stack application.

## 1. Prerequisites
-   A VPS (Ubuntu/Debian recommended) with a Public IP.
-   **Docker** & **Docker Compose** installed on the server.
-   A domain name (e.g., `mysite.com`) pointing to your server's IP (DNS A Record).

## 2. Server Setup (One-Time)

### Step 1: Copy Files
Transfer the entire project to the server.

**Structure on Server:**
```text
/your-server-path/
├── adgc2nn-backend/       <-- Copy this whole folder (It contains app/, docker files, etc)
│   ├── docker-compose.prod.yml
│   ├── app/
│   ├── nginx/
│   └── ...
└── adgc2nn-frontend/      <-- Copy this whole folder too
```

*Note: The `docker-compose.prod.yml` expects `../adgc2nn-frontend` to exist relative to the backend folder.*

### Step 2: Configure Domain & SSL
Before running anything, update the config files with your real domain.

**A. Edit `init-letsencrypt.sh`**
```bash
domains=(mysite.com www.mysite.com) # Your Domain
email="you@email.com"               # Your Email
staging=0                           # Set to 0 for Production Certs
```

**B. Edit `nginx/conf.d/app.conf`**
Replace `example.com` with your real domain in both `server` blocks.

### Step 3: Initialize Certificates
Run this script to get your first SSL certificate. It automates the complex "Bootstrapping" process.
```bash
chmod +x init-letsencrypt.sh
./init-letsencrypt.sh
```
*Success? You should see "Requesting Let's Encrypt certificate... Success!"*

## 3. Deployment & Updates

### Start / Update the Server
Whenever you update code or want to restart, run this command:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### Stop the Server
```bash
docker compose -f docker-compose.prod.yml down
```

## 4. Maintenance

### Monitoring Logs
See what's happening live:
```bash
docker compose -f docker-compose.prod.yml logs -f
```

### SSL Renewals
Certbot is configured to **automatically renew** your certificates. You don't need to do anything. It runs a check every 12 hours.

## 5. Troubleshooting checklist

| Issue | Check |
| :--- | :--- |
| **"Bad Gateway (502)"** | Is the Frontend container running? (`docker ps`) |
| **"Connection Refused"** | Is the Backend running? Did Nginx fail to start? |
| **"Not Secure" in Browser** | Did `init-letsencrypt.sh` run successfully? Is port 443 open? |
