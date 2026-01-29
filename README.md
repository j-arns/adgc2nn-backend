# Saturation Vapor Pressure Estimator - Backend API

This repository contains the high-performance **FastAPI** backend for the Saturation Vapor Pressure Estimator. It serves an Adaptive-Depth Graph Convolutional Neural Network (adGC2NN) model to predict vapor pressure ($P_{sat}$) at 298 K from SMILES strings.

## 🏗 Architecture

The system is designed as a decoupled microservice architecture, optimized for scientific computing and high-throughput inference.

### Core Stack

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+) for high-performance, async API handling.
* **Data Validation:** [Pydantic](https://docs.pydantic.dev/) ensures strict type safety for inputs/outputs.
* **ML Engine:** PyTorch/NumPy based inference engine (`adGC2NN`).
* **Data Processing:** Pandas for CSV/batch handling.

### API Design Pattern

The backend exposes RESTful endpoints consumed by a **Next.js** frontend via **Server Actions**.

1. **`/predict`**: Optimized for single-item, real-time inference with low latency.
2. **`/batch_predict`**: A **polymorphic endpoint** that handles:
    * `multipart/form-data`: For large CSV/TXT file uploads, returning a streamed CSV response.

## 📂 Directory Structure

```text
adgc2nn-api/
├── .dockerignore                 # Files to exclude from Docker builds.
├── Dockerfile                    # Blueprint for building the backend container image.
├── docker-compose.yml            # Services for LOCAL development.
├── docker-compose.prod.yml       # Services for PRODUCTION (adds HTTPS).
├── init-letsencrypt.sh           # Helper script for SSL certificates.
├── requirements.txt              # List of Python dependencies.
├── README.md                     # General project overview.
├── LICENSE                       # Project license.
│
├── app/                          # Main Application Source Code
│   ├── main.py                   # App Entry Point.
│   │
│   ├── api/v1/                   # API Route definitions
│   │   ├── router.py             # Central router.
│   │   └── endpoints/
│   │       └── prediction.py     # Logic for /predict and /batch_predict.
│   │
│   ├── core/
│   │   └── config.py             # Global settings.
│   │
│   ├── schemas/
│   │   └── prediction.py         # Pydantic models (Data validation).
│   │
│   └── services/                 # Business Logic & ML Engine
│       ├── engine.py             # Orchestrates predictions.
│       ├── application_functions.py  # Core ML Logic & Graph conversion.
│       ├── confined_model_weights.pth  # Model weights (C/H/O).
│       ├── broad_model_weights.pth     # Model weights (Heteroatoms).
│       ├── conf_normalizer.pickle      # Output Scaler (Confined).
│       └── broad_normalizer.pickle     # Output Scaler (Broad).
│
├── nginx/                        # Nginx Configuration
│   ├── nginx.conf                # Main Nginx global settings.
│   ├── app.local.conf            # Config for LOCALHOST.
│   └── conf.d/
│       └── app.conf              # Config for PRODUCTION (HTTPS).
│
└── certbot/                      # Shared volume for SSL certificates.
```

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Virtual Environment (recommended)
* **Docker** & **Docker Compose** (for containerized local run)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-org/adgc2nn-backend.git
   cd adgc2nn-backend
   ```

2. **Create virtual environment (Optional for Docker):**

   ```bash
   python -m venv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies (Local Python only):**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**

   **Option A: Python (Direct)**
   ```bash
   uvicorn app.main:app --reload --port 3001
   ```

   **Option B: Docker (Simulate Production)**
   This runs the full stack (Frontend + Backend + Nginx) locally without SSL.
   ```bash
   docker compose up --build
   ```
   Access the app at http://localhost.

## 📡 API Reference

### 1. Single Prediction

**Endpoint:** `POST /predict`

**Request:**

```json
{
  "smiles": "CCO"
}
```

**Response:**

```json
{
  "smiles": "CCO",
  "prediction": 6931.07,
  "model": "adGC2NN-confined"
}
```

### 2. Batch Prediction (JSON)

**Endpoint:** `POST /batch_predict`
**Header:** `Content-Type: application/json`

**Request:**

```json
{
  "smiles_list": ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
}
```

### 3. Batch Prediction (File)

**Endpoint:** `POST /batch_predict`
**Header:** `Content-Type: multipart/form-data`

**Input:** A `.csv` or `.txt` file containing one SMILES string per line.

**Response:** Returns a downloadable `text/csv` stream containing columns: `SMILES`, `Prediction[Pa]`, `log10-Prediction[Pa]`, `Model`.

