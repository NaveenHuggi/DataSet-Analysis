# Dataset Analyser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203-FF6B35?style=for-the-badge)

**A full-stack AI-powered data science platform.**  
Upload a dataset, get an instant deep analysis, chat with an AI ML mentor, compare datasets, generate training scripts, and run live Python — all in one clean interface.

</div>

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Running the Application](#running-the-application)
6. [First Launch](#first-launch)
7. [API Reference](#api-reference)
8. [AI Models](#ai-models)
9. [Dependencies](#dependencies)
10. [Security Notes](#security-notes)
11. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
DataSet-Analysis/
├── backend/
│   ├── main.py              # FastAPI server — all API endpoints
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Auto-created on first run (stores Groq API key)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Full React application
│   │   ├── index.css        # Design system (light mode, CSS variables)
│   │   └── main.jsx         # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md
```

---

## Features

### Analysis Tab

- **Multi-format upload** — CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet
- **8-metric stat tiles** — rows, columns, missing %, duplicates, memory, numeric count, categorical count, file format
- **Column type breakdown** — colour-coded chips listing all numeric and categorical columns
- **Profiling report** — on-demand `ydata-profiling` HTML report rendered inline with a download button
- **Feature Engineering Ideas** — AI-generated suggestions (6–8 ideas with Python snippets) via Llama 3.3-70B
- **Phase 1 AI Analysis** — structured deep analysis: Dataset Summary, Data Health Check, Problem Type, Recommended Models

### Compare Tab

- Side-by-side overview table (rows, columns, missing, duplicates, memory)
- Column overlap analysis — common, only-in-A, only-in-B
- Per-column numeric comparison (mean, std, missing count) for up to 12 shared columns

### AI Mentor Tab

- **Streaming chat** — real-time token streaming via Server-Sent Events
- **Model selector** — Llama 3.1 8B, Llama 3.3 70B, Gemma 2 9B, Mixtral 8x7B
- **Training Script Generator** — pick an algorithm + target column → generates a complete `.py` script (copy/download)
- **Chat export** — download the full conversation as a Markdown file
- **Rich Markdown rendering** — code blocks, tables, headings, bullet lists

### Code Sandbox Tab

- **Live Python execution** with `df`, `pd`, `np`, `plt`, `sns` pre-loaded
- **8 quick-insert snippets** — head, describe, info, missing values, heatmap, distribution, value counts, pairplot
- **Matplotlib chart rendering** — plots render inline as images
- `Ctrl+Enter` to run · `Tab` for indentation

---

## Prerequisites

Before you begin, you need the following installed on your system.

### 1. Python 3.10 or higher

**Check if already installed:**
```bash
python --version
# or on some systems:
python3 --version
```

**Install Python:**
- **Windows**: Download from [python.org/downloads](https://www.python.org/downloads/). During installation, tick **"Add Python to PATH"**.
- **macOS**: `brew install python` (requires [Homebrew](https://brew.sh)) or download from python.org.
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install python3 python3-pip python3-venv`

### 2. Node.js 18 or higher

**Check if already installed:**
```bash
node --version
npm --version
```

**Install Node.js:**
- **All platforms**: Download the LTS installer from [nodejs.org](https://nodejs.org). npm is included automatically.
- **macOS** (alternative): `brew install node`
- **Linux (Ubuntu/Debian)**: `sudo apt install nodejs npm`

### 3. Git

**Check if already installed:**
```bash
git --version
```

**Install Git:**
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **macOS**: `brew install git` or `xcode-select --install`
- **Linux**: `sudo apt install git`

### 4. A Groq API Key (free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`) — you will paste it into the app on first launch

---

## Installation & Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/NaveenHuggi/DataSet-Analysis.git
cd DataSet-Analysis
```

> If you downloaded a ZIP instead, extract it and open a terminal inside the extracted folder.

---

### Step 2 — Set up the Python backend

It is strongly recommended to use a **virtual environment** so the project packages don't conflict with your system Python.

#### Windows (PowerShell or Command Prompt)

```powershell
cd backend

# Create a virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Your prompt will now show (venv) — install the dependencies
pip install -r requirements.txt
```

#### macOS / Linux

```bash
cd backend

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt will now show (venv) — install the dependencies
pip install -r requirements.txt
```

> **Note:** Installing `ydata-profiling` may take a few minutes — it has several large dependencies. This is normal.

---

### Step 3 — Install the frontend dependencies

Open a **new, separate terminal** (keep the backend terminal open) and run:

```bash
cd DataSet-Analysis/frontend

npm install
```

This downloads all React and Vite packages into the `frontend/node_modules/` folder. It only needs to be run once.

---

## Running the Application

You need **two terminals running at the same time** — one for the backend, one for the frontend.

### Terminal 1 — Start the Backend

```bash
# Navigate to the backend folder
cd DataSet-Analysis/backend

# Activate the virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Start the server
uvicorn main:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

The `--reload` flag means the server automatically restarts when you edit `main.py`.

---

### Terminal 2 — Start the Frontend

```bash
# Navigate to the frontend folder
cd DataSet-Analysis/frontend

# Start the development server
npm run dev
```

You should see output like:

```
  VITE v5.x.x  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

Open **http://localhost:5173** in your browser.

---

## First Launch

1. **Setup Screen** — On first visit, you will be prompted to enter your Groq API key.
2. **Enter the key** — Paste your `gsk_...` key and click **Continue**. The key is validated live against the Groq API.
3. **Key is saved** — The key is written to `backend/.env` and never shown again. Future launches skip the setup screen.
4. **Upload a dataset** — Go to the **Analysis** tab and drag-and-drop a CSV, Excel, JSON, or Parquet file.
5. **Run Phase 1 Analysis** — Click the "Run Phase 1 Analysis" button to get an AI-powered breakdown, then switch to **AI Mentor** to continue the conversation.

---

### Stopping the Application

- Press `Ctrl + C` in each terminal to stop the backend and frontend servers.

### Restarting After a Reboot

Every time you open the project again, you just need to re-run the two start commands:

```bash
# Terminal 1
cd DataSet-Analysis/backend
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
uvicorn main:app --reload

# Terminal 2
cd DataSet-Analysis/frontend
npm run dev
```

---

## API Reference

Interactive Swagger docs are available at `http://localhost:8000/docs` while the backend is running.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | API key status and loaded dataset metadata |
| `GET` | `/api/models` | List of available Groq models |
| `POST` | `/api/setup` | Validate and persist the Groq API key |
| `POST` | `/api/upload` | Upload and parse a dataset file |
| `POST` | `/api/profile` | Generate a ydata-profiling HTML report |
| `POST` | `/api/compare/upload` | Upload a second dataset for comparison |
| `GET` | `/api/compare/stats` | Run side-by-side comparison of both datasets |
| `POST` | `/api/analyze` | Phase 1 AI analysis (Llama 3.3-70B) |
| `POST` | `/api/features` | AI feature engineering suggestions |
| `POST` | `/api/training-script` | Generate a complete Python training script |
| `POST` | `/api/chat` | Streaming mentor chat via SSE |
| `POST` | `/api/chat/clear` | Reset chat history |
| `POST` | `/api/execute` | Execute Python code in the sandbox |

---

## AI Models

| Model | Used for |
|---|---|
| `llama-3.3-70b-versatile` | Phase 1 analysis, feature suggestions, training script generation |
| `llama-3.1-8b-instant` *(default)* | Streaming mentor chat |
| `gemma2-9b-it` | Mentor chat (user-selectable) |
| `mixtral-8x7b-32768` | Mentor chat — long-context conversations |

---

## Dependencies

### Backend

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn[standard]` | ASGI server |
| `python-multipart` | File upload handling |
| `python-dotenv` | `.env` file management |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `openpyxl` | Excel file support |
| `pyarrow` | Parquet file support |
| `groq` | Groq API client |
| `ydata-profiling` | Automated HTML EDA reports |
| `matplotlib` | Chart generation (sandbox) |
| `seaborn` | Statistical plots (sandbox) |

### Frontend

| Package | Purpose |
|---|---|
| `react`, `react-dom` | UI framework |
| `vite` | Build tool and dev server |
| `lucide-react` | Icon library |
| `react-markdown` | Markdown rendering in chat |

---

## Security Notes

> **Local use only.**

The Code Sandbox executes arbitrary Python via `exec()` directly on the machine running the backend. This is safe for personal local use but **the app must not be deployed publicly** without replacing the sandbox with a properly isolated environment (e.g. a Docker container with CPU/memory limits and no network access).

The `.gitignore` already excludes `backend/.env` so your API key is never committed to version control.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python: command not found` | Use `python3` instead, or reinstall Python and tick "Add to PATH" |
| `npm: command not found` | Install Node.js from [nodejs.org](https://nodejs.org) and restart the terminal |
| `venv\Scripts\activate` fails on Windows | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell first |
| `pip install` fails on `ydata-profiling` | Ensure you are using Python 3.10+ and the virtual environment is active |
| Sandbox shows "Cannot reach backend" | Start uvicorn in Terminal 1: `uvicorn main:app --reload` |
| API key rejected on setup | Ensure the key starts with `gsk_` and was copied in full from console.groq.com |
| Profiling report takes a long time | Expected for large files — `ydata-profiling` does a full statistical scan |
| Excel upload fails with `ModuleNotFoundError` | Run `pip install openpyxl` with the venv active |
| Parquet upload fails with `ModuleNotFoundError` | Run `pip install pyarrow` with the venv active |
| Backend status dot stays orange | The backend is still starting — wait a few seconds; it turns green automatically |
| Port 8000 already in use | Run `uvicorn main:app --reload --port 8001` and update `frontend/src/App.jsx` line `const API = 'http://localhost:8001/api'` |
| Port 5173 already in use | Vite will automatically try 5174, 5175, etc. Check the terminal output for the actual URL |

---

## Acknowledgements

- [Groq](https://groq.com) — ultra-fast inference on open-weight models
- [Meta AI](https://ai.meta.com) — Llama 3 model family
- [Mistral AI](https://mistral.ai) — Mixtral model
- [Google DeepMind](https://deepmind.google) — Gemma 2 model
- [FastAPI](https://fastapi.tiangolo.com) — Python API framework
- [Vite](https://vitejs.dev) — frontend build tooling
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) — automated EDA reports

---

## License

MIT
