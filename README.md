# 🤖 AI Data Scientist & ML Mentor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An intelligent, locally-run Streamlit app that acts as your personal ML mentor.**  
Upload any CSV → Get instant deep EDA → Chat with an AI mentor → Run live Python code — all in one place.

</div>

---

## ✨ Features

### 📊 Tab 1 — Dataset Analysis
- **Instant inline EDA charts** rendered the moment you upload a CSV:
  - 📋 Overview metrics — rows, columns, missing cells (with smart `<0.1%` handling), duplicates
  - 🔴 Missing values bar chart — color-coded by severity (HIGH / MEDIUM / LOW)
  - 📊 Numeric distributions — histogram grid with mean overlay for every numeric column
  - 🔗 Correlation heatmap — full Pearson matrix + top-10 correlated pairs table
  - 🏷️ Categorical value counts — top-12 values per categorical column
  - 🗂️ Data types — pie chart of column dtype breakdown
- **Full profiling report** with 3-tier engine fallback — always works:
  1. `ydata-profiling` *(richest report)*
  2. `sweetviz` *(lighter alternative)*
  3. **Custom pandas HTML report** *(zero extra deps — always succeeds)*
- **⬇️ Download HTML Report** button — save the full profiling report locally
- **🤖 AI Deep Analysis** — send the dataset profile to Llama 3.3-70B for a structured report covering Dataset Summary, Data Health Check, Problem Type, and Recommended Models

### 💬 Tab 2 — AI Mentor Chat
- Conversational, context-aware chat powered by **Llama 3.1-8B-Instant** (fast, streaming)
- Remembers your full dataset profile across the conversation
- Guides you step-by-step through preprocessing and model training
- Never dumps a complete script — paces you through logical chunks

### 🧪 Tab 3 — Code Sandbox
- Full **Python execution environment** inside the app — like a single Jupyter cell runner
- Pre-loaded variables available in every run:

  | Variable | What it is |
  |---|---|
  | `df` | Your uploaded DataFrame |
  | `pd` | pandas |
  | `np` | numpy |
  | `plt` | matplotlib.pyplot |
  | `sns` | seaborn |
  | `px` | plotly.express |
  | `go` | plotly.graph_objects |
  | `sklearn` | scikit-learn (all submodules) |
  | `scipy`, `stats` | SciPy and scipy.stats |

- Captures `stdout` (print statements)
- Renders `matplotlib` and `plotly` figures inline
- Auto-evaluates the last expression (so `df.head()` shows a table — no `print()` needed)

### 🔑 API Key Persistence
- Enter your Groq API key **once** — it is validated and saved to a `.env` file in the project folder
- On every subsequent launch the key is loaded automatically — **the setup screen never appears again**

---

## 🖥️ Screenshots

> *Upload a CSV and the app immediately renders all EDA charts, lets you generate a full profiling report, chat with your AI mentor, and run Python code live.*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) *(takes ~30 seconds to create)*

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/ai-data-scientist-mentor.git
cd ai-data-scientist-mentor
```

### 2 — One-click setup (Windows)

Double-click **`setup.bat`** — it will:
1. Create an isolated virtual environment (`venv/`)
2. Upgrade pip
3. Install all dependencies

```
setup.bat
```

### 3 — Launch the app

Double-click **`run_app.bat`** — it activates the venv and opens the app at `http://localhost:8501`.

```
run_app.bat
```

### Manual setup (Mac / Linux)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-data-scientist-mentor/
│
├── app.py                  ← Main Streamlit application (all logic)
├── requirements.txt        ← Pinned, conflict-free dependencies
│
├── setup.bat               ← Windows: create venv + install deps (run once)
├── run_app.bat             ← Windows: activate venv + launch app (run every time)
│
├── .env                    ← Auto-created on first launch — stores your Groq API key
│                             (git-ignored, never committed)
├── .gitignore              ← Excludes venv/, .env, __pycache__
└── README.md
```

---

## ⚙️ Configuration

### Groq API Key

The app validates and saves your key to `.env` on first launch. You can also set it manually:

```bash
# .env
GROQ_API_KEY=gsk_your_key_here
```

> **Note:** `.env` is git-ignored by default — your key will never be accidentally committed.

### AI Models Used

| Phase | Model | Purpose |
|---|---|---|
| Dataset Analysis | `llama-3.3-70b-versatile` | Deep analytical report (Phase 1) |
| Mentor Chat | `llama-3.1-8b-instant` | Fast streaming conversation (Phase 2) |

Both are free-tier models on Groq's API.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas`, `numpy` | Data manipulation |
| `groq` | Groq API client (Llama 3 models) |
| `python-dotenv` | `.env` file management |
| `ydata-profiling` | Rich HTML profiling report *(optional)* |
| `sweetviz` | Lighter profiling fallback *(optional)* |
| `matplotlib`, `seaborn` | Inline EDA charts |
| `plotly` | Interactive charts in sandbox |
| `scikit-learn`, `scipy` | ML libraries available in the sandbox |

> **Profiling fallback chain:** `ydata-profiling` → `sweetviz` → custom pandas HTML (always works, zero extra deps)

---

## 💡 Usage Guide

### Uploading a dataset
1. Go to the **📊 Dataset Analysis** tab
2. Upload any `.csv` file
3. All EDA charts render instantly below the uploader
4. Click **"🔬 Generate Full Profiling Report"** for a detailed HTML report
5. Click **"🚀 Generate AI Analysis"** to get the LLM-powered breakdown

### Using the AI Mentor Chat
1. Complete the AI Analysis on the Dataset Analysis tab first
2. Switch to **💬 AI Mentor Chat**
3. The AI will ask which model you want to build — pick one and it guides you step-by-step

### Using the Code Sandbox
1. Upload a dataset first (makes `df` available)
2. Switch to **🧪 Code Sandbox**
3. Write or paste Python code — `df`, `pd`, `np`, `plt`, `sklearn`, etc. are all pre-loaded
4. Click **▶ Run Code**

```python
# Example sandbox code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

---

## 🔒 Security Notes

- The code sandbox uses Python's built-in `exec()` to run user-provided code.
- **This is safe for local use** — the app is designed as a personal, locally-run tool.
- ⚠️ **Do not deploy this app publicly** (e.g., on Streamlit Cloud or any server) without removing or sandboxing the code execution feature.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'groq'` | Run `setup.bat` first, or `pip install -r requirements.txt` inside the venv |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` inside the activated venv |
| `ydata-profiling not available` | The app auto-falls back to sweetviz → pandas. No action needed |
| API key keeps being asked | Check that `.env` was created in the project folder. It needs `GROQ_API_KEY=gsk_...` |
| Dependency conflicts with other projects | Always use `run_app.bat` (runs inside the isolated venv) — never your global Python |
| Charts look oversized | Resize your browser window — all charts scale with container width |

---

## 🤝 Contributing

Pull requests are welcome! Here's how to get started:

```bash
git clone https://github.com/your-username/ai-data-scientist-mentor.git
cd ai-data-scientist-mentor
python -m venv venv
venv\Scripts\activate        # or: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Please open an issue first for major changes.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — for blazing-fast inference on Llama 3 models
- [Meta AI](https://ai.meta.com) — for the open-weight Llama 3 model family
- [Streamlit](https://streamlit.io) — for making data apps ridiculously easy to build
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) — for the rich EDA reports
- [sweetviz](https://github.com/fbdesignpro/sweetviz) — for the lightweight profiling fallback

---

<div align="center">
  Made with ❤️ · <a href="https://console.groq.com">Get a free Groq API key →</a>
</div>
