# Project Documentation: AdGC2NN Backend API

## 1. Executive Summary

The **AdGC2NN Backend API** is a high-performance serving layer for the Adaptive-Depth Graph Convolutional Neural Network (adGC2NN). It provides real-time inference for estimating the **Saturation Vapor Pressure ($P_{sat}$)** of chemical compounds at 298 K based on their molecular structure (SMILES).

The system is built as a microservice using **FastAPI** (Python 3.10+), utilizing **PyTorch Geometric** for graph neural network operations and **RDKit** for cheminformatics preprocessing. It supports both single-instance and batch predictions, dynamically selecting between a "Confined" and "Broad" model architecture based on molecular composition.

---

## 2. System Architecture

The project adheres to a service-oriented architecture, decoupling the API interface from the Model Inference Engine.

### 2.1 Technology Stack
*   **API Framework:** FastAPI (Asynchronous, Type-safe via Pydantic).
*   **Inference Engine:** PyTorch & PyTorch Geometric (GNN implementations).
*   **Cheminformatics:** RDKit (Molecular parsing, validation, and featurization).
*   **Data Processing:** Pandas & NumPy (Batch handling).
*   **Containerization:** Docker & Docker Compose.

### 2.2 Data Flow Pipeline
1.  **Request Handling:** The `v1/router.py` receives HTTP POST requests containing SMILES strings.
2.  **Input Validation:** Pydantic models in `schemas/` validate the JSON structure.
3.  **Engine Dispatch:** `services/engine.py` orchestrates the inference workflow:
    *   **Validity Check:** Uses RDKit (`check_molecule`) to verify chemical validity.
    *   **Model Selection:** Routes the input to either the **Confined** (C/H/O non-aromatic) or **Broad** (Heteroatoms + aromatic) pipeline.
4.  **Graph Transformation:** `services/application_functions.py` converts the SMILES string into a graph structure (Nodes=Atoms, Edges=Bonds).
5.  **Inference:** The `AdGC2NN` PyTorch module executes the forward pass.
6.  **Normalization:** The raw model output (Log-space) is inverse-transformed to Pascals ($Pa$) via pickle-serialized Scalers.
7.  **Response:** The API returns the computed pressure in JSON or CSV format.

---

## 3. Directory Structure & Key Components

### Root Directory
*   **`main.py`**: Application entry point. Configures the FastAPI instance, CORS middleware, and mounts routers.
*   **`Dockerfile` / `docker-compose.yml`**: Defines the container environment, exposing port **3001**.
*   **`requirements.txt`**: Strict dependency pinning, including `torch-geometric`, `rdkit`, and `fastapi`.

### `app/services/` (Core Logic)
This directory contains the business logic and ML artifacts.

#### `engine.py` (Orchestrator)
The `AdGC2NNEngine` class implements the Singleton pattern to load model weights (`.pth`) and scalers (`.pickle`) into memory at startup. It exposes two main methods:
*   `predict_single(smiles: str)`: Low-latency prediction for distinct lookups.
*   `predict_batch(smiles_list: List[str])`: Optimized batch processing. It separates inputs into "Confined" and "Broad" batches, processes them in vectorized operations, and recombines the results to maintain input order.

#### `application_functions.py` (Data Pipeline & Model Definition)

---
Enter description here.
---

## 5. Infrastructure: Common Foundation

The project utilizes a containerized microservice architecture to ensuring consistency across development and production environments. This approach solves the "it works on my machine" problem, particularly for complex dependencies like RDKit.

### 5.1 Docker Containerization

The backend utilizes a custom `Dockerfile` optimized for security and size:

*   **Base Image:** `python:3.12-slim`. This is a minimal Debian-based image that reduces the attack surface and download size compared to the full Python image.
*   **System Dependencies:** We explicitly install system-level libraries (`libxrender1`, `libxext6`) required by **RDKit** for molecular rendering and calculation. These are often missing in standard Python environments.
*   **Security (Non-Root User):**
    *   By default, Docker containers run as `root`.
    *   Our Dockerfile creates a specific user `appuser` (UID 1000) and group `appgroup`.
    *   All application code is owned by this user.
    *   **Why?** If the application is compromised, the attacker only gains control of a restricted user, preventing them from modifying the container's system files or escaping to the host.
*   **Networking:** The container exposes port `3001` internally.

---

## 6. Chapter A: Local Deployment (Development)

This setup is designed for developers running the code on their own laptop. It mimics production but processes are simplified (no HTTPS, no real domains).

### 6.1 Configuration
*   **File:** `docker-compose.yml`
*   **Nginx Config:** `nginx/app.local.conf`
*   **Port:** Exposes **HTTP** on port `80`.

### 6.2 Architecture Diagram (Local)
1.  **Backend (`backend`)**: Hidden inside the network (Port 3001).
2.  **Frontend (`frontend`)**: Connects to backend via `http://backend:3001`.
3.  **Nginx**: Maps `localhost:80` -> `frontend:3000`.

### 6.3 How to Run
```bash
# Build and start the cluster locally
docker compose up --build
```
Access the application at `http://localhost`.

---

## 7. Chapter B: Production Deployment (Server)

This setup is designed for a public Linux server. It adds security layers (SSL/TLS) and separates concerns for stability.

### 7.1 Configuration
*   **File:** `docker-compose.prod.yml`
*   **Nginx Config:** `nginx/conf.d/app.conf`
*   **Ports:** Exposes **HTTP** (80) and **HTTPS** (443).
*   **Services Added:** `certbot` (Automatically handles SSL certificates from Let's Encrypt).

### 7.2 Architecture Diagram (Production)

The production setup introduces an automatic HTTPS redirection and certificate management loop.

1.  **Nginx (Entry Point):**
    *   **Port 80:** Listens for HTTP traffic and **Permamently Redirects (301)** it to HTTPS (Port 443). Also handles the ACME challenge for Certbot.
    *   **Port 443:** Decrypts SSL traffic. Checks certificates mounted from the `certbot` volume. Forwards valid traffic to `frontend:3000`.

2.  **Certbot (Sidecar):**
    *   A separate container that talks to Let's Encrypt servers.
    *   It places the specialized "Challenge" files in a shared folder (`/var/www/certbot`) which Nginx serves to prove domain ownership.
    *   It renews certificates automatically.

### 7.3 Deployment Steps

**Step 1: Configure Domain**
Open `nginx/conf.d/app.conf` and `init-letsencrypt.sh`. Replace `example.com` with your actual domain name.

**Step 2: Initialize SSL (First Time Only)**
We have a helper script to safely request the initial certificates.
```bash
chmod +x init-letsencrypt.sh
sudo ./init-letsencrypt.sh
```

**Step 3: Run the Application**
Use the production compose file.
```bash
docker compose -f docker-compose.prod.yml up -d --build
```
*   `-f docker-compose.prod.yml`: Tells docker to use the production config.
*   `-d`: Detached mode (runs in background so it stays alive when you disconnect).

