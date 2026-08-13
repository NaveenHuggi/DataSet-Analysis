import os
import io
import contextlib
import traceback
import base64
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv, set_key

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

ANALYSIS_MODEL    = "llama-3.3-70b-versatile"
CHAT_MODEL_DEFAULT = "llama-3.1-8b-instant"

AVAILABLE_MODELS = [
    {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B",  "note": "Fastest"},
    {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B", "note": "Most Powerful"},
    {"id": "gemma2-9b-it",            "label": "Gemma 2 9B",    "note": "Balanced"},
    {"id": "mixtral-8x7b-32768",      "label": "Mixtral 8x7B",  "note": "Long Context"},
]

SYSTEM_PROMPT = """You are an Automated AI Data Scientist and Machine Learning Mentor. Your purpose is to analyze dataset profiles and guide the user step-by-step through preprocessing and model training.

You operate in two distinct phases:

### Phase 1: The Initial Analysis
When the user first uploads or pastes a dataset profile, provide a structured report following exactly these four sections:
1. **Dataset Summary**: Explain in 2-3 simple, jargon-free sentences what this dataset represents and its likely real-world domain.
2. **Data Health Check**: Highlight immediate issues the user needs to fix before training (missing values, skewed distributions, high cardinality, leakage risks).
3. **Problem Type**: State clearly whether this is Classification, Regression, Clustering, Time-Series, etc., and why.
4. **Recommended Models**: Recommend 2-3 specific ML models best suited for this data with brief justification.

At the end of Phase 1, ask the user: "Which of these models would you like to build first?"

### Phase 2: The Interactive Walkthrough
Once the user selects a path, transition into a highly conversational, step-by-step mentor.
- **Pacing**: Never output a massive script. Break it into logical, verifiable chunks.
- **Code Quality**: Provide clean, commented Python code.
- **Verification**: Always end a step by asking if it ran successfully before proceeding."""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dataset Analyser API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State (single-user local app) ──────────────────────────────────────
GLOBAL_STATE: dict = {
    "df":               None,
    "df_compare":       None,
    "profile_text":     "",
    "compare_profile":  "",
    "chat_history":     [],
    "filename":         "",
    "filename_compare": "",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(content))
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    elif ext == ".json":
        return pd.read_json(io.BytesIO(content))
    elif ext == ".parquet":
        return pd.read_parquet(io.BytesIO(content))
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

def _build_profile_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    missing = df.isnull().sum()
    missing_str = missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values."
    profile  = f"--- Dataset Info ---\n{buf.getvalue()}\n\n"
    profile += f"--- Missing Values ---\n{missing_str}\n\n"
    profile += f"--- Statistical Summary ---\n{df.describe(include='all').to_string()}\n\n"
    profile += f"--- Sample Data (first 5 rows) ---\n{df.head().to_string()}\n\n"
    return profile

def _dataset_stats(df: pd.DataFrame, filename: str) -> dict:
    numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols         = df.select_dtypes(exclude=np.number).columns.tolist()
    missing_total    = int(df.isnull().sum().sum())
    total_cells      = len(df) * len(df.columns)
    missing_pct      = round(missing_total / total_cells * 100, 2) if total_cells else 0
    return {
        "filename":           filename,
        "rows":               len(df),
        "columns":            len(df.columns),
        "columns_list":       df.columns.tolist(),
        "numeric_columns":    numeric_cols,
        "categorical_columns": cat_cols,
        "missing_cells":      missing_total,
        "missing_pct":        missing_pct,
        "duplicate_rows":     int(df.duplicated().sum()),
        "memory_kb":          round(df.memory_usage(deep=True).sum() / 1024, 1),
    }

def _get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Groq API key not configured. Please set it in the setup screen.")
    return Groq(api_key=key)

# ── Status & Setup ────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    key = os.getenv("GROQ_API_KEY", "").strip()
    df  = GLOBAL_STATE["df"]
    return {
        "api_key_set": bool(key),
        "dataset_loaded": df is not None,
        "filename": GLOBAL_STATE["filename"],
        "rows": len(df) if df is not None else 0,
        "columns": len(df.columns) if df is not None else 0,
    }

@app.get("/api/models")
def get_models():
    return {"models": AVAILABLE_MODELS}

class SetupRequest(BaseModel):
    api_key: str

@app.post("/api/setup")
def setup_api(req: SetupRequest):
    key = req.api_key.strip()
    try:
        client = Groq(api_key=key)
        client.chat.completions.create(
            model=CHAT_MODEL_DEFAULT,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        ENV_FILE.touch(exist_ok=True)
        set_key(str(ENV_FILE), "GROQ_API_KEY", key)
        os.environ["GROQ_API_KEY"] = key
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Upload & Profile ──────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = _read_file(content, file.filename)
        GLOBAL_STATE["df"]           = df
        GLOBAL_STATE["filename"]     = file.filename
        GLOBAL_STATE["profile_text"] = _build_profile_text(df)
        GLOBAL_STATE["chat_history"] = []
        return _dataset_stats(df, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/profile")
def get_profile():
    df = GLOBAL_STATE["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    try:
        from ydata_profiling import ProfileReport
        report = ProfileReport(df, title="Dataset Profiling Report", minimal=True,
                               explorative=False, progress_bar=False)
        return {"html": report.to_html()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Dataset Comparison ────────────────────────────────────────────────────────
@app.post("/api/compare/upload")
async def upload_compare(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = _read_file(content, file.filename)
        GLOBAL_STATE["df_compare"]       = df
        GLOBAL_STATE["filename_compare"] = file.filename
        GLOBAL_STATE["compare_profile"]  = _build_profile_text(df)
        return _dataset_stats(df, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/compare/stats")
def get_compare_stats():
    df1 = GLOBAL_STATE["df"]
    df2 = GLOBAL_STATE["df_compare"]
    if df1 is None or df2 is None:
        raise HTTPException(status_code=400, detail="Both datasets must be uploaded first.")

    cols1, cols2   = set(df1.columns), set(df2.columns)
    common_cols    = sorted(cols1 & cols2)
    num1 = df1.select_dtypes(include=np.number)
    num2 = df2.select_dtypes(include=np.number)
    common_numeric = sorted(set(num1.columns) & set(num2.columns))

    numeric_comparison = {}
    for col in common_numeric[:12]:
        numeric_comparison[col] = {
            "mean_1":    round(float(df1[col].mean()), 4),
            "mean_2":    round(float(df2[col].mean()), 4),
            "std_1":     round(float(df1[col].std()),  4),
            "std_2":     round(float(df2[col].std()),  4),
            "missing_1": int(df1[col].isnull().sum()),
            "missing_2": int(df2[col].isnull().sum()),
        }

    return {
        "dataset1":         _dataset_stats(df1, GLOBAL_STATE["filename"]),
        "dataset2":         _dataset_stats(df2, GLOBAL_STATE["filename_compare"]),
        "common_columns":   common_cols,
        "only_in_1":        sorted(cols1 - cols2),
        "only_in_2":        sorted(cols2 - cols1),
        "numeric_comparison": numeric_comparison,
    }

# ── AI Analysis ───────────────────────────────────────────────────────────────
@app.post("/api/analyze")
def analyze_dataset():
    profile_text = GLOBAL_STATE["profile_text"]
    if not profile_text:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is my dataset profile. Please provide the Phase 1 Initial Analysis.\n\n{profile_text}"}
            ],
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
        GLOBAL_STATE["chat_history"] = [{"role": "assistant", "content": analysis}]
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Feature Engineering Suggestions ──────────────────────────────────────────
@app.post("/api/features")
def suggest_features():
    profile_text = GLOBAL_STATE["profile_text"]
    if not profile_text:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert data scientist specializing in feature engineering for machine learning."},
                {"role": "user", "content": f"""Based on this dataset profile, suggest 6-8 concrete, actionable feature engineering ideas.

For each feature, provide in this EXACT format:
**Feature Name**: [name]
**How**: [brief python expression or formula, e.g. df['col_a'] / df['col_b']]
**Why**: [one sentence explaining the ML benefit]

Be specific and practical. Only suggest features that make sense given the actual column names and data types in the profile.

Dataset Profile:
{profile_text}"""}
            ],
            temperature=0.4,
        )
        return {"suggestions": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Training Script Generation ────────────────────────────────────────────────
class TrainingScriptRequest(BaseModel):
    model_type: str
    target_column: str = ""

@app.post("/api/training-script")
def generate_training_script(req: TrainingScriptRequest):
    profile_text = GLOBAL_STATE["profile_text"]
    if not profile_text:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    client = _get_groq_client()
    
    # Include recent chat context
    recent_chat = "\n".join(
        [f"{m['role'].upper()}: {m['content'][:400]}" for m in GLOBAL_STATE["chat_history"][-6:]]
    )
    target_hint = f"Target column: {req.target_column}" if req.target_column else "Infer the target column from context."

    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ML engineer. Generate complete, production-quality, well-commented Python training scripts."},
                {"role": "user", "content": f"""Generate a COMPLETE, standalone Python training script for a **{req.model_type}** model.

{target_hint}

Dataset Profile:
{profile_text}

Recent mentor conversation context:
{recent_chat}

Requirements:
- All imports at the top
- Load data from 'data.csv' (or whatever filename is in the profile)
- Complete preprocessing pipeline (handle missing values, encode categoricals, scale numerics)
- Train/test split
- Model training with sensible hyperparameters
- Multiple evaluation metrics
- Save the trained model with joblib
- Print a summary at the end

Return ONLY the Python code. Use inline comments to explain each major step."""}
            ],
            temperature=0.15,
            max_tokens=2500,
        )
        return {"script": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str = CHAT_MODEL_DEFAULT

@app.post("/api/chat/clear")
def clear_chat():
    GLOBAL_STATE["chat_history"] = []
    return {"status": "cleared"}

@app.post("/api/chat")
def chat_mentor(req: ChatRequest):
    client = _get_groq_client()

    GLOBAL_STATE["chat_history"].append({"role": "user", "content": req.message})

    system = SYSTEM_PROMPT
    if GLOBAL_STATE["profile_text"]:
        system += f"\n\nContext — User's Dataset Profile:\n{GLOBAL_STATE['profile_text']}"

    llm_messages = [{"role": "system", "content": system}]
    llm_messages += [{"role": m["role"], "content": m["content"]} for m in GLOBAL_STATE["chat_history"]]

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model=req.model,
                messages=llm_messages,
                stream=True,
                temperature=0.5,
            )
            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            GLOBAL_STATE["chat_history"].append({"role": "assistant", "content": full_response})
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── Code Sandbox ──────────────────────────────────────────────────────────────
class ExecuteRequest(BaseModel):
    code: str

@app.post("/api/execute")
def execute_code(req: ExecuteRequest):
    df = GLOBAL_STATE.get("df") or pd.DataFrame()

    _scope = {
        "df": df, "pd": pd, "np": np,
        "plt": plt, "sns": sns,
        "__builtins__": __builtins__,
    }

    stdout_capture = io.StringIO()
    _error = None
    plt.close("all")

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(req.code, "<sandbox>", "exec"), _scope)
    except Exception:
        _error = traceback.format_exc()

    stdout_text = stdout_capture.getvalue()

    # Auto-eval last expression if no output
    if not _error and not stdout_text:
        lines = [l for l in req.code.strip().splitlines() if l.strip()]
        if lines:
            try:
                result = eval(compile(lines[-1], "<sandbox_eval>", "eval"), _scope)
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        stdout_text = result.head(50).to_string()
                    elif isinstance(result, pd.Series):
                        stdout_text = result.head(50).to_frame().to_string()
                    else:
                        stdout_text = repr(result)
            except Exception:
                pass

    figs = [plt.figure(n) for n in plt.get_fignums()]
    base64_images = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        base64_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close(fig)

    return {"stdout": stdout_text, "error": _error, "images": base64_images}
