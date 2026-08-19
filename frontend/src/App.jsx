import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Database, MessageSquare, Terminal, Upload, BarChart2, Send,
  Play, RefreshCw, Download, Cpu, FileText, AlertCircle,
  CheckCircle, ChevronRight, Trash2, GitCompare, Lightbulb,
  Code, ChevronDown, Copy, X, Zap
} from 'lucide-react';

// Relative path — Vite dev-server proxy (vite.config.js) forwards /api/* to
// http://localhost:8000. No CORS headers needed.
const API = '/api';

// ── Tiny helpers ──────────────────────────────────────────────────────────────
function fmt(n) { return n?.toLocaleString() ?? '—'; }

async function apiFetch(path, opts = {}) {
  let res;
  try {
    res = await fetch(`${API}${path}`, opts);
  } catch (networkErr) {
    throw new Error(
      'Cannot reach the backend server.\n' +
      'Make sure uvicorn is running:\n  cd backend && uvicorn main:app --reload'
    );
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
  return data;
}

// Live backend status (polls every 10 s)
function useBackendStatus() {
  const [online, setOnline] = useState(null); // null = checking
  useEffect(() => {
    let cancelled = false;
    const check = () =>
      fetch(`${API}/status`, { signal: AbortSignal.timeout(3000) })
        .then(() => { if (!cancelled) setOnline(true); })
        .catch(() => { if (!cancelled) setOnline(false); });
    check();
    const id = setInterval(check, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  return online;
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => {});
}

function downloadText(text, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  a.download = filename;
  a.click();
}

function exportChatMarkdown(messages) {
  const lines = ['# AI Mentor Chat Export\n'];
  messages.forEach(m => {
    lines.push(`## ${m.role === 'user' ? 'You' : 'AI Mentor'}\n\n${m.content}\n`);
  });
  downloadText(lines.join('\n---\n\n'), 'chat_export.md');
}

// ── Loading Spinner ───────────────────────────────────────────────────────────
function Spinner({ size = 14 }) {
  return <RefreshCw size={size} className="spin" />;
}

// ════════════════════════════════════════════════════════════════════
// SETUP SCREEN
// ════════════════════════════════════════════════════════════════════
function SetupScreen({ onSuccess }) {
  const [key, setKey]       = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      await apiFetch('/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key.trim() })
      });
      onSuccess();
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const features = [
    { icon: <Upload size={13} />,      label: 'Multi-format Upload' },
    { icon: <BarChart2 size={13} />,   label: 'EDA Profiling' },
    { icon: <MessageSquare size={13}/>, label: 'AI Mentor Chat' },
    { icon: <Terminal size={13} />,    label: 'Code Sandbox' },
    { icon: <GitCompare size={13} />,  label: 'Dataset Compare' },
    { icon: <Code size={13} />,        label: 'Script Generator' },
  ];

  return (
    <div className="setup-screen">
      <div className="setup-panel">
        <div className="setup-logo-row">
          <div className="setup-logo-icon"><BarChart2 size={22} color="white" /></div>
          <div>
            <h1>Dataset<span> Analyser</span></h1>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>AI-Powered Data Science Platform</p>
          </div>
        </div>

        <p className="setup-desc">
          Connect your Groq API Key to unlock AI-powered dataset analysis, an interactive ML mentor, and a live Python sandbox.
        </p>

        <div className="setup-features">
          {features.map(f => (
            <div className="setup-feature-chip" key={f.label}>{f.icon} {f.label}</div>
          ))}
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label className="input-label">Groq API Key</label>
            <input className="input-field" type="password" placeholder="gsk_..." value={key}
              onChange={e => setKey(e.target.value)} required />
          </div>
          {error && (
            <div className="alert alert-error mb-3">
              <AlertCircle size={14} /> {error}
            </div>
          )}
          <button type="submit" className="btn btn-primary w-full" disabled={loading}>
            {loading ? <><Spinner /> Validating...</> : <>Continue <ChevronRight size={15} /></>}
          </button>
        </form>

        <p className="setup-hint">
          Get a free key at <a href="https://console.groq.com" target="_blank" rel="noreferrer">console.groq.com</a> — saved locally, never committed.
        </p>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// SIDEBAR
// ════════════════════════════════════════════════════════════════════
function Sidebar({ activeTab, setActiveTab, hasDataset, backendOnline }) {
  const navItems = [
    { id: 'analysis', icon: <Database size={16} />,    label: 'Analysis' },
    { id: 'compare',  icon: <GitCompare size={16} />,  label: 'Compare' },
    { id: 'chat',     icon: <MessageSquare size={16}/>, label: 'AI Mentor' },
    { id: 'sandbox',  icon: <Terminal size={16} />,    label: 'Code Sandbox' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon"><BarChart2 size={18} color="white" /></div>
        <div className="logo-text">
          <h1>DatasetAI</h1>
          <span>Analysis Platform</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {navItems.map(item => (
          <button key={item.id}
            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => setActiveTab(item.id)}
          >
            {item.icon} {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-row">
          <span
            className="status-dot"
            style={{
              background: backendOnline === null ? '#d97706'
                        : backendOnline ? '#059669' : '#e11d48',
              boxShadow: backendOnline === null ? '0 0 0 2px rgba(217,119,6,0.2)'
                       : backendOnline ? '0 0 0 2px rgba(5,150,105,0.2)'
                       : '0 0 0 2px rgba(225,29,72,0.2)',
            }}
          />
          {backendOnline === null ? 'Connecting…'
         : backendOnline ? 'Backend Online'
         : 'Backend Offline'}
        </div>
        {hasDataset && (
          <div className="status-row" style={{ marginTop: 6, color: 'var(--green)' }}>
            <CheckCircle size={11} style={{ marginRight: 5 }} /> Dataset Loaded
          </div>
        )}
        <div className="sidebar-model-tag">Powered by Groq</div>
      </div>
    </aside>
  );
}

// ════════════════════════════════════════════════════════════════════
// ANALYSIS TAB  (all state passed in as props — never lost)
// ════════════════════════════════════════════════════════════════════
function AnalysisTab({
  file, setFile, stats, setStats,
  profileHtml, setProfileHtml,
  analysisDone, setAnalysisDone,
  featureSuggestions, setFeatureSuggestions,
  onAnalysisComplete
}) {
  const [dragging,    setDragging]    = useState(false);
  const [uploading,   setUploading]   = useState(false);
  const [profiling,   setProfiling]   = useState(false);
  const [analyzing,   setAnalyzing]   = useState(false);
  const [featureLoading, setFeatureLoading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const inputRef = useRef();

  const ACCEPT = '.csv,.xlsx,.xls,.json,.parquet';

  const processFile = async (selected) => {
    if (!selected) return;
    const ext = selected.name.split('.').pop().toLowerCase();
    if (!['csv','xlsx','xls','json','parquet'].includes(ext)) {
      setUploadError('Unsupported file type. Please upload CSV, Excel, JSON, or Parquet.');
      return;
    }
    setFile(selected); setUploadError(''); setUploading(true);
    const formData = new FormData();
    formData.append('file', selected);
    try {
      const data = await apiFetch('/upload', { method: 'POST', body: formData });
      setStats(data);
      setProfileHtml(null);
      setAnalysisDone(false);
      setFeatureSuggestions('');
    } catch (err) { setUploadError(err.message); }
    finally { setUploading(false); }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    processFile(e.dataTransfer.files[0]);
  };

  const handleGenerateProfile = async () => {
    setProfiling(true);
    try {
      const data = await apiFetch('/profile', { method: 'POST' });
      setProfileHtml(data.html);
    } catch (err) { alert('Profiling failed: ' + err.message); }
    finally { setProfiling(false); }
  };

  const handleRunAI = async () => {
    setAnalyzing(true);
    try {
      const data = await apiFetch('/analyze', { method: 'POST' });
      setAnalysisDone(true);
      onAnalysisComplete(data.analysis);
    } catch (err) { alert('AI Analysis failed: ' + err.message); }
    finally { setAnalyzing(false); }
  };

  const handleFeatures = async () => {
    setFeatureLoading(true);
    try {
      const data = await apiFetch('/features', { method: 'POST' });
      setFeatureSuggestions(data.suggestions);
    } catch (err) { alert('Feature generation failed: ' + err.message); }
    finally { setFeatureLoading(false); }
  };

  const statTiles = stats ? [
    { label: 'Rows',     value: fmt(stats.rows) },
    { label: 'Columns',  value: fmt(stats.columns) },
    { label: 'Missing',  value: `${stats.missing_pct}%` },
    { label: 'Duplicates', value: fmt(stats.duplicate_rows) },
    { label: 'Memory',   value: `${stats.memory_kb} KB` },
    { label: 'Numerics', value: fmt(stats.numeric_columns?.length) },
    { label: 'Categoricals', value: fmt(stats.categorical_columns?.length) },
    { label: 'Format',   value: stats.filename?.split('.').pop()?.toUpperCase() ?? '—' },
  ] : [];

  return (
    <div>
      {/* Upload */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title"><Upload size={16} /> Upload Dataset</h3>
          <span className="badge badge-gray">CSV · Excel · JSON · Parquet</span>
        </div>

        {!stats ? (
          <div
            className={`drop-zone ${dragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="drop-icon"><Upload size={22} /></div>
            <h3>{uploading ? 'Uploading...' : 'Drop your file here'}</h3>
            <p>Supports CSV, Excel (.xlsx), JSON, and Parquet · or click to browse</p>
          </div>
        ) : (
          <div>
            <div className="alert alert-success mb-3">
              <CheckCircle size={14} />
              <span><strong>{stats.filename}</strong> loaded — {fmt(stats.rows)} rows × {stats.columns} columns</span>
            </div>
            <button className="btn btn-ghost" onClick={() => inputRef.current.click()}>
              <Upload size={14} /> Replace File
            </button>
          </div>
        )}
        <input ref={inputRef} type="file" accept={ACCEPT} style={{ display: 'none' }}
          onChange={e => processFile(e.target.files[0])} />
        {uploadError && <div className="alert alert-error mt-3"><AlertCircle size={13} /> {uploadError}</div>}
      </div>

      {/* Stats Grid */}
      {stats && (
        <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginTop: 16 }}>
          {statTiles.map(t => (
            <div className="stat-tile" key={t.label}>
              <div className="value">{t.value}</div>
              <div className="label">{t.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Column Lists */}
      {stats && (
        <div className="card" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          <div>
            <p className="card-title" style={{ marginBottom: 10 }}>
              <span style={{ color: 'var(--accent)' }}>⬡</span> Numeric Columns ({stats.numeric_columns?.length})
            </p>
            <div className="col-chip-wrap">
              {stats.numeric_columns?.map(c => <span className="col-chip numeric" key={c}>{c}</span>)}
            </div>
          </div>
          <div>
            <p className="card-title" style={{ marginBottom: 10 }}>
              <span style={{ color: 'var(--accent-2)' }}>⬡</span> Categorical Columns ({stats.categorical_columns?.length})
            </p>
            <div className="col-chip-wrap">
              {stats.categorical_columns?.map(c => <span className="col-chip categorical" key={c}>{c}</span>)}
            </div>
          </div>
        </div>
      )}

      {/* Profiling */}
      {stats && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title"><FileText size={16} /> Profiling Report</h3>
            <div className="flex gap-2">
              {profileHtml && (
                <a
                  href={`data:text/html;charset=utf-8,${encodeURIComponent(profileHtml)}`}
                  download="profiling_report.html" className="btn btn-ghost"
                  style={{ textDecoration: 'none' }}
                >
                  <Download size={13} /> Download
                </a>
              )}
              <button className="btn btn-ghost" onClick={handleGenerateProfile} disabled={profiling}>
                {profiling ? <><Spinner size={13} /> Generating...</> : 'Generate Report'}
              </button>
            </div>
          </div>
          {profileHtml ? (
            <iframe srcDoc={profileHtml}
              style={{ width: '100%', height: 580, border: '1px solid var(--border-subtle)', borderRadius: 8, background: 'white' }} />
          ) : (
            <p className="card-desc" style={{ margin: 0 }}>
              Click "Generate Report" to run a full ydata-profiling analysis. This may take 30–60 seconds for larger files.
            </p>
          )}
        </div>
      )}

      {/* Feature Engineering */}
      {stats && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title"><Lightbulb size={16} /> Feature Engineering Ideas</h3>
            <button className="btn btn-ghost" onClick={handleFeatures} disabled={featureLoading}>
              {featureLoading ? <><Spinner size={13} /> Generating...</> : 'Get AI Suggestions'}
            </button>
          </div>
          {featureSuggestions ? (
            <div className="feature-markdown">
              <ReactMarkdown>{featureSuggestions}</ReactMarkdown>
            </div>
          ) : (
            <p className="card-desc" style={{ margin: 0 }}>
              Ask the AI for concrete, dataset-specific feature engineering ideas with Python code snippets.
            </p>
          )}
        </div>
      )}

      {/* AI Analysis */}
      {stats && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title"><Cpu size={16} /> AI Deep Analysis</h3>
            {analysisDone && <span className="badge badge-green"><CheckCircle size={11} /> Complete</span>}
          </div>
          {analysisDone ? (
            <div className="alert alert-success">
              <CheckCircle size={14} />
              Phase 1 analysis complete. Switch to <strong>AI Mentor</strong> to continue.
            </div>
          ) : (
            <>
              <p className="card-desc">
                Run the Phase 1 analysis: the AI will identify the problem type, data health issues,
                and recommend models tailored to your dataset.
              </p>
              <button className="btn btn-primary" onClick={handleRunAI} disabled={analyzing}>
                {analyzing ? <><Spinner size={13} /> Analyzing...</> : <><Zap size={14} /> Run Phase 1 Analysis</>}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// COMPARE TAB
// ════════════════════════════════════════════════════════════════════
function CompareTab({ hasDataset, compareStats, setCompareStats, compareFile, setCompareFile }) {
  const [compareUploading, setCompareUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const inputRef = useRef();

  const processCompareFile = async (selected) => {
    if (!selected) return;
    setCompareFile(selected); setUploadError(''); setCompareUploading(true);
    const formData = new FormData();
    formData.append('file', selected);
    try {
      await apiFetch('/compare/upload', { method: 'POST', body: formData });
    } catch (err) { setUploadError(err.message); }
    finally { setCompareUploading(false); }
  };

  const handleCompare = async () => {
    setLoading(true);
    try {
      const data = await apiFetch('/compare/stats');
      setCompareStats(data);
    } catch (err) { alert(err.message); }
    finally { setLoading(false); }
  };

  const d1 = compareStats?.dataset1;
  const d2 = compareStats?.dataset2;

  return (
    <div>
      {!hasDataset && (
        <div className="alert alert-info mb-4">
          <AlertCircle size={14} /> Upload a primary dataset in the <strong>Analysis</strong> tab first.
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3 className="card-title"><GitCompare size={16} /> Compare Datasets</h3>
        </div>
        <p className="card-desc">Upload a second dataset to compare statistics, columns, and distributions side-by-side.</p>

        {!compareFile ? (
          <div className="drop-zone" onClick={() => inputRef.current.click()}>
            <div className="drop-icon"><Upload size={20} /></div>
            <h3>{compareUploading ? 'Uploading...' : 'Upload Comparison Dataset'}</h3>
            <p>Same formats supported: CSV, Excel, JSON, Parquet</p>
          </div>
        ) : (
          <div className="alert alert-success mb-3">
            <CheckCircle size={14} /> <strong>{compareFile.name}</strong> ready for comparison
          </div>
        )}
        <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls,.json,.parquet"
          style={{ display: 'none' }} onChange={e => processCompareFile(e.target.files[0])} />
        {uploadError && <div className="alert alert-error mt-3"><AlertCircle size={13} /> {uploadError}</div>}

        {compareFile && hasDataset && (
          <button className="btn btn-primary mt-3" onClick={handleCompare} disabled={loading}>
            {loading ? <><Spinner size={13} /> Comparing...</> : <><GitCompare size={14} /> Run Comparison</>}
          </button>
        )}
      </div>

      {compareStats && (
        <>
          {/* Overview */}
          <div className="card">
            <h3 className="card-title mb-4"><BarChart2 size={16} /> Overview</h3>
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>{d1.filename}</th>
                  <th>{d2.filename}</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Rows',       fmt(d1.rows),       fmt(d2.rows)],
                  ['Columns',    fmt(d1.columns),    fmt(d2.columns)],
                  ['Numeric',    fmt(d1.numeric_columns?.length), fmt(d2.numeric_columns?.length)],
                  ['Categorical',fmt(d1.categorical_columns?.length), fmt(d2.categorical_columns?.length)],
                  ['Missing Cells', fmt(d1.missing_cells), fmt(d2.missing_cells)],
                  ['Missing %',  `${d1.missing_pct}%`, `${d2.missing_pct}%`],
                  ['Duplicates', fmt(d1.duplicate_rows), fmt(d2.duplicate_rows)],
                  ['Memory',     `${d1.memory_kb} KB`, `${d2.memory_kb} KB`],
                ].map(([label, v1, v2]) => (
                  <tr key={label}>
                    <td className="compare-label">{label}</td>
                    <td>{v1}</td>
                    <td>{v2}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Column Overlap */}
          <div className="card" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20 }}>
            <div>
              <p className="compare-section-title">Common Columns ({compareStats.common_columns?.length})</p>
              <div className="col-chip-wrap">
                {compareStats.common_columns?.map(c => <span className="col-chip numeric" key={c}>{c}</span>)}
              </div>
            </div>
            <div>
              <p className="compare-section-title">Only in {d1.filename}</p>
              <div className="col-chip-wrap">
                {compareStats.only_in_1?.length > 0
                  ? compareStats.only_in_1.map(c => <span className="col-chip categorical" key={c}>{c}</span>)
                  : <span className="text-muted">None</span>}
              </div>
            </div>
            <div>
              <p className="compare-section-title">Only in {d2.filename}</p>
              <div className="col-chip-wrap">
                {compareStats.only_in_2?.length > 0
                  ? compareStats.only_in_2.map(c => <span className="col-chip categorical" key={c}>{c}</span>)
                  : <span className="text-muted">None</span>}
              </div>
            </div>
          </div>

          {/* Numeric Comparison */}
          {Object.keys(compareStats.numeric_comparison || {}).length > 0 && (
            <div className="card">
              <h3 className="card-title mb-4"><BarChart2 size={16} /> Numeric Column Comparison</h3>
              <table className="compare-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Mean ({d1.filename})</th>
                    <th>Mean ({d2.filename})</th>
                    <th>Std ({d1.filename})</th>
                    <th>Std ({d2.filename})</th>
                    <th>Missing 1</th>
                    <th>Missing 2</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(compareStats.numeric_comparison).map(([col, v]) => (
                    <tr key={col}>
                      <td className="compare-label">{col}</td>
                      <td>{v.mean_1}</td>
                      <td>{v.mean_2}</td>
                      <td>{v.std_1}</td>
                      <td>{v.std_2}</td>
                      <td>{v.missing_1}</td>
                      <td>{v.missing_2}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// CHAT TAB
// ════════════════════════════════════════════════════════════════════
function ChatTab({ messages, setMessages, initialAnalysis, availableModels }) {
  const [input,    setInput]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [model,    setModel]    = useState('llama-3.1-8b-instant');
  const [showScript, setShowScript] = useState(false);
  const [scriptLoading, setScriptLoading] = useState(false);
  const [script, setScript]     = useState('');
  const [scriptModelType, setScriptModelType] = useState('Random Forest');
  const [scriptTarget, setScriptTarget] = useState('');
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  const SCRIPT_MODELS = [
    'Random Forest', 'XGBoost', 'Logistic Regression', 'Linear Regression',
    'SVM', 'Neural Network (MLPClassifier)', 'K-Nearest Neighbors', 'Decision Tree',
  ];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (initialAnalysis && messages.length === 0) {
      setMessages([{ role: 'assistant', content: initialAnalysis }]);
    }
  }, [initialAnalysis]);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return;
    setInput(''); setLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: text.trim() }]);
    setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true }]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text.trim(), model })
      });
      if (!res.ok) throw new Error('Chat request failed');

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let accum = '', buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.delta) {
              accum += payload.delta;
              setMessages(prev => {
                const copy = [...prev];
                copy[copy.length - 1] = { role: 'assistant', content: accum, streaming: true };
                return copy;
              });
            }
          } catch (_) {}
        }
      }
      setMessages(prev => {
        const copy = [...prev];
        copy[copy.length - 1] = { role: 'assistant', content: accum, streaming: false };
        return copy;
      });
    } catch (err) {
      setMessages(prev => {
        const copy = [...prev];
        copy[copy.length - 1] = { role: 'assistant', content: `Error: ${err.message}`, streaming: false };
        return copy;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, model, setMessages]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  const clearChat = async () => {
    try { await apiFetch('/chat/clear', { method: 'POST' }); } catch (_) {}
    setMessages([]);
  };

  const handleGenerateScript = async () => {
    setScriptLoading(true);
    try {
      const data = await apiFetch('/training-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: scriptModelType, target_column: scriptTarget })
      });
      setScript(data.script);
    } catch (err) { alert(err.message); }
    finally { setScriptLoading(false); }
  };

  const currentModelLabel = availableModels.find(m => m.id === model)?.label ?? model;

  return (
    <div className="chat-layout" style={{ height: 'calc(100vh - 58px)' }}>

      {/* Toolbar */}
      <div className="chat-toolbar">
        <div className="flex items-center gap-2">
          <span className="chat-toolbar-label">Model</span>
          <div className="select-wrapper">
            <select className="model-select" value={model} onChange={e => setModel(e.target.value)}>
              {availableModels.map(m => (
                <option key={m.id} value={m.id}>{m.label} — {m.note}</option>
              ))}
            </select>
            <ChevronDown size={13} className="select-chevron" />
          </div>
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <span className="chat-toolbar-label">{messages.length} messages</span>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => setShowScript(s => !s)}>
            <Code size={13} /> Generate Script
          </button>
          {messages.length > 0 && (
            <button className="btn btn-ghost btn-sm" onClick={() => exportChatMarkdown(messages)}>
              <Download size={13} /> Export
            </button>
          )}
          {messages.length > 0 && (
            <button className="btn btn-danger btn-sm" onClick={clearChat}>
              <Trash2 size={13} /> Clear
            </button>
          )}
        </div>
      </div>

      {/* Script Panel */}
      {showScript && (
        <div className="script-panel">
          <div className="script-panel-header">
            <span className="flex items-center gap-2"><Code size={15} /> Generate Training Script</span>
            <button className="icon-btn" onClick={() => setShowScript(false)}><X size={15} /></button>
          </div>
          <div className="script-panel-controls">
            <div className="flex items-center gap-3" style={{ flex: 1 }}>
              <div className="input-group" style={{ margin: 0, flex: 1 }}>
                <label className="input-label">Algorithm</label>
                <div className="select-wrapper">
                  <select className="model-select" value={scriptModelType} onChange={e => setScriptModelType(e.target.value)}>
                    {SCRIPT_MODELS.map(m => <option key={m}>{m}</option>)}
                  </select>
                  <ChevronDown size={12} className="select-chevron" />
                </div>
              </div>
              <div className="input-group" style={{ margin: 0, flex: 1 }}>
                <label className="input-label">Target Column (optional)</label>
                <input className="input-field" placeholder="e.g. price, survived, target…"
                  value={scriptTarget} onChange={e => setScriptTarget(e.target.value)} />
              </div>
            </div>
            <button className="btn btn-primary" onClick={handleGenerateScript} disabled={scriptLoading}>
              {scriptLoading ? <><Spinner size={13} /> Generating...</> : <><Zap size={13} /> Generate</>}
            </button>
          </div>
          {script && (
            <div className="script-output">
              <div className="script-output-header">
                <span>{scriptModelType} Training Script</span>
                <div className="flex gap-2">
                  <button className="btn btn-ghost btn-sm" onClick={() => copyToClipboard(script)}>
                    <Copy size={12} /> Copy
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={() => downloadText(script, 'train.py')}>
                    <Download size={12} /> Download .py
                  </button>
                </div>
              </div>
              <pre className="script-code">{script}</pre>
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon"><MessageSquare size={26} /></div>
            <h3>Start a conversation</h3>
            <p>Run Phase 1 Analysis from the Dataset Analysis tab for an instant AI breakdown, or ask your mentor anything directly.</p>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`message-row ${msg.role === 'user' ? 'user' : ''}`}>
              <div className={`avatar ${msg.role === 'user' ? 'avatar-user' : 'avatar-ai'}`}>
                {msg.role === 'user' ? 'U' : 'AI'}
              </div>
              <div className={`message-bubble ${msg.role === 'user' ? 'user' : 'ai'}`}>
                {msg.role === 'user' ? (
                  <span>{msg.content}</span>
                ) : (
                  <>
                    <ReactMarkdown
                    components={{
                      // Tables — use the styled compare-table class
                      table: ({ node, ...p }) => <table className="compare-table" style={{ marginTop: 10, marginBottom: 10 }} {...p} />,
                      thead: ({ node, ...p }) => <thead {...p} />,
                      tbody: ({ node, ...p }) => <tbody {...p} />,
                      tr:    ({ node, ...p }) => <tr {...p} />,
                      th:    ({ node, ...p }) => <th {...p} />,
                      td:    ({ node, ...p }) => <td {...p} />,
                      // Headings
                      h1: ({ node, ...p }) => <h1 style={{ fontSize: 17, fontWeight: 700, marginTop: 18, marginBottom: 8, color: 'var(--text-primary)', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 6 }} {...p} />,
                      h2: ({ node, ...p }) => <h2 style={{ fontSize: 15, fontWeight: 700, marginTop: 16, marginBottom: 6, color: 'var(--text-primary)' }} {...p} />,
                      h3: ({ node, ...p }) => <h3 style={{ fontSize: 13.5, fontWeight: 600, marginTop: 12, marginBottom: 4, color: 'var(--accent-dark)' }} {...p} />,
                      // Code
                      code: ({ node, inline, ...p }) => inline
                        ? <code style={{ fontFamily: 'monospace', fontSize: 12.5, background: 'rgba(99,102,241,0.08)', padding: '2px 5px', borderRadius: 4, color: 'var(--accent-dark)' }} {...p} />
                        : <pre style={{ fontFamily: 'monospace', fontSize: 12.5, background: '#1e1e2e', color: '#cdd6f4', padding: '12px 14px', borderRadius: 8, overflowX: 'auto', marginTop: 8, marginBottom: 8, lineHeight: 1.6 }}><code {...p} /></pre>,
                      // Lists
                      ul: ({ node, ...p }) => <ul style={{ paddingLeft: 20, marginTop: 6, marginBottom: 6, display: 'flex', flexDirection: 'column', gap: 4 }} {...p} />,
                      ol: ({ node, ...p }) => <ol style={{ paddingLeft: 20, marginTop: 6, marginBottom: 6, display: 'flex', flexDirection: 'column', gap: 4 }} {...p} />,
                      li: ({ node, ...p }) => <li style={{ color: 'var(--text-secondary)', fontSize: 13.5, lineHeight: 1.6 }} {...p} />,
                      // Paragraph
                      p:  ({ node, ...p }) => <p style={{ marginBottom: 8, lineHeight: 1.7, color: 'var(--text-secondary)', fontSize: 13.5 }} {...p} />,
                      // Strong / em
                      strong: ({ node, ...p }) => <strong style={{ fontWeight: 600, color: 'var(--text-primary)' }} {...p} />,
                      em:     ({ node, ...p }) => <em style={{ color: 'var(--text-muted)', fontStyle: 'italic' }} {...p} />,
                    }}
                  >{msg.content}</ReactMarkdown>
                    {msg.streaming && (
                      <span style={{ display: 'inline-block', width: 8, height: 14, background: 'var(--accent)', borderRadius: 2, marginLeft: 3, verticalAlign: 'text-bottom', animation: 'typingBounce 1s ease-in-out infinite' }} />
                    )}
                  </>
                )}
              </div>
            </div>
          ))
        )}
        {loading && messages[messages.length - 1]?.streaming !== true && (
          <div className="message-row">
            <div className="avatar avatar-ai">AI</div>
            <div className="message-bubble ai">
              <div className="typing-indicator">
                <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-row">
          <textarea ref={inputRef} className="chat-input-field"
            placeholder={`Ask ${currentModelLabel}... (Shift+Enter for new line)`}
            value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown} disabled={loading} rows={1}
            onInput={e => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
            }}
          />
          <button className="chat-send-btn" onClick={() => sendMessage(input)} disabled={loading || !input.trim()}>
            <Send size={15} />
          </button>
        </div>
        <p className="chat-helper-text">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// SANDBOX TAB
// ════════════════════════════════════════════════════════════════════
const SNIPPETS = [
  { label: 'df.head()',          code: 'df.head()' },
  { label: 'df.describe()',      code: 'df.describe()' },
  { label: 'df.info()',          code: 'import io\nbuf = io.StringIO()\ndf.info(buf=buf)\nprint(buf.getvalue())' },
  { label: 'Missing values',     code: 'df.isnull().sum().sort_values(ascending=False)' },
  { label: 'Correlation heatmap',code: 'import seaborn as sns\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(10, 7))\nsns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)\nfig.tight_layout()\nplt.show()' },
  { label: 'Distribution plot',  code: 'import matplotlib.pyplot as plt\ncol = df.select_dtypes(include="number").columns[0]\nplt.figure(figsize=(8, 4))\nplt.hist(df[col].dropna(), bins=30, color="#6366f1", edgecolor="white")\nplt.title(f"Distribution of {col}")\nplt.tight_layout()\nplt.show()' },
  { label: 'Value counts',       code: 'col = df.select_dtypes(include="object").columns[0]\ndf[col].value_counts().head(15)' },
  { label: 'Pairplot',           code: 'import seaborn as sns\nimport matplotlib.pyplot as plt\nnumeric_cols = df.select_dtypes(include="number").columns[:5]\nsns.pairplot(df[numeric_cols].dropna())\nplt.tight_layout()\nplt.show()' },
];

function SandboxTab({ hasDataset }) {
  const [code,    setCode]    = useState('print(df.shape)\ndf.dtypes');
  const [output,  setOutput]  = useState(null);
  const [running, setRunning] = useState(false);

  const handleRun = async () => {
    setRunning(true);
    try {
      const data = await apiFetch('/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      setOutput(data);
    } catch (err) {
      setOutput({ error: err.message, stdout: '', images: [] });
    } finally { setRunning(false); }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); handleRun(); }
    if (e.key === 'Tab') {
      e.preventDefault();
      const s = e.target.selectionStart, end = e.target.selectionEnd;
      const newCode = code.substring(0, s) + '    ' + code.substring(end);
      setCode(newCode);
      setTimeout(() => { e.target.selectionStart = e.target.selectionEnd = s + 4; }, 0);
    }
  };

  return (
    <div>
      {!hasDataset && (
        <div className="alert alert-info mb-4">
          <AlertCircle size={14} style={{ flexShrink: 0 }} />
          Upload a dataset in the <strong>Analysis</strong> tab first so <code>df</code> is available.
        </div>
      )}

      {/* Backend offline warning */}
      <div id="sandbox-offline-banner" style={{ display: 'none' }}>
        <div className="alert alert-error mb-4">
          <AlertCircle size={14} style={{ flexShrink: 0 }} />
          <span>
            <strong>Backend not running.</strong> Open a terminal in the <code>backend/</code> folder and run:<br />
            <code style={{ display: 'block', marginTop: 4 }}>uvicorn main:app --reload</code>
          </span>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header">
          <h3 className="card-title"><Terminal size={16} /> Quick Snippets</h3>
          <span className="text-muted" style={{ fontSize: 12 }}>Click to insert</span>
        </div>
        <div className="snippet-grid">
          {SNIPPETS.map(s => (
            <button key={s.label} className="snippet-chip" onClick={() => setCode(s.code)}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title"><Terminal size={16} /> Python Editor</h3>
          <div className="flex gap-2">
            <span className="badge badge-gray" style={{ fontSize: 11 }}>
              df · pd · np · plt · sns pre-loaded
            </span>
            {output && (
              <button className="btn btn-ghost btn-sm" onClick={() => setOutput(null)}>
                <Trash2 size={12} /> Clear
              </button>
            )}
            <button className="btn btn-primary" onClick={handleRun} disabled={running}>
              {running ? <><Spinner size={13} /> Running...</> : <><Play size={13} /> Run (Ctrl+Enter)</>}
            </button>
          </div>
        </div>

        <textarea className="sandbox-editor mb-3" value={code}
          onChange={e => setCode(e.target.value)} onKeyDown={handleKeyDown}
          spellCheck={false} style={{ minHeight: 250 }} />

        {output && (
          <div className="sandbox-output">
            <div className="sandbox-output-header">
              <div className="sandbox-output-header-left">
                <Terminal size={13} /> <span>Output</span>
              </div>
              {output.error
                ? <span className="sandbox-status-err">Error</span>
                : <span className="sandbox-status-ok">Success</span>}
            </div>
            <div className="sandbox-output-body">
              {output.error   && <pre className="error-pre">{output.error}</pre>}
              {output.stdout  && <pre>{output.stdout}</pre>}
              {output.images?.map((b64, i) => (
                <img key={i} src={`data:image/png;base64,${b64}`} alt={`plot-${i}`}
                  style={{ maxWidth: '100%', marginTop: 12, borderRadius: 8, display: 'block' }} />
              ))}
              {!output.error && !output.stdout && !output.images?.length && (
                <pre style={{ color: '#7a7fa8' }}>Code executed with no output.</pre>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// APP ROOT
// ════════════════════════════════════════════════════════════════════
const TAB_META = {
  analysis: { title: 'Dataset Analysis',  subtitle: 'Upload, profile, and get an AI-powered breakdown of your dataset.' },
  compare:  { title: 'Compare Datasets',  subtitle: 'Upload a second file to compare statistics and column overlap side-by-side.' },
  chat:     { title: 'AI Mentor',          subtitle: 'Stream a conversation with your personal data science mentor.' },
  sandbox:  { title: 'Code Sandbox',       subtitle: 'Execute Python against your dataset in a live local environment.' },
};

export default function App() {
  const [setupComplete,  setSetupComplete]  = useState(null);
  const [activeTab,      setActiveTab]      = useState('analysis');
  const backendOnline = useBackendStatus();

  // ── Dataset state (lifted — persists across tab switches) ──────────
  const [file,              setFile]              = useState(null);
  const [stats,             setStats]             = useState(null);
  const [profileHtml,       setProfileHtml]       = useState(null);
  const [analysisDone,      setAnalysisDone]      = useState(false);
  const [featureSuggestions,setFeatureSuggestions]= useState('');
  const [initialAnalysis,   setInitialAnalysis]   = useState('');

  // ── Compare state ──────────────────────────────────────────────────
  const [compareFile,  setCompareFile]  = useState(null);
  const [compareStats, setCompareStats] = useState(null);

  // ── Chat state ─────────────────────────────────────────────────────
  const [messages,       setMessages]       = useState([]);
  const [availableModels, setAvailableModels] = useState([
    { id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B', note: 'Fastest' },
    { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B', note: 'Most Powerful' },
  ]);

  // ── Init ───────────────────────────────────────────────────────────
  useEffect(() => {
    // Check API key + restore minimal dataset state if backend still has it
    fetch(`${API}/status`, { signal: AbortSignal.timeout(4000) })
      .then(r => r.json())
      .then(d => {
        setSetupComplete(d.api_key_set);
        if (d.dataset_loaded) {
          setStats({ filename: d.filename, rows: d.rows, columns: d.columns });
        }
      })
      .catch(() => {
        // Backend not up yet — show setup screen so user knows something is wrong
        setSetupComplete(false);
      });

    // Fetch available models (best-effort)
    fetch(`${API}/models`, { signal: AbortSignal.timeout(4000) })
      .then(r => r.json())
      .then(d => { if (d.models) setAvailableModels(d.models); })
      .catch(() => {});
  }, []);

  const handleAnalysisComplete = (analysis) => {
    setInitialAnalysis(analysis);
    setMessages([{ role: 'assistant', content: analysis }]);
    setActiveTab('chat');
  };

  if (setupComplete === null) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        <Spinner size={20} />
      </div>
    );
  }

  if (!setupComplete) {
    return <SetupScreen onSuccess={() => setSetupComplete(true)} />;
  }

  const meta = TAB_META[activeTab];

  return (
    <div className="app-shell">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab}
        hasDataset={!!stats} backendOnline={backendOnline} />

      <div className="main-content">
        {/* Top Bar */}
        <div className="topbar">
          <div>
            <div className="topbar-title">{meta.title}</div>
          </div>
          <div className="topbar-subtitle">{meta.subtitle}</div>
          <div className="flex gap-2">
            {stats && <span className="badge badge-green"><CheckCircle size={11} /> {stats.filename ?? 'Dataset Loaded'}</span>}
            {backendOnline === false && (
              <span className="badge" style={{ background: 'rgba(225,29,72,0.08)', color: 'var(--red)', borderColor: 'rgba(225,29,72,0.2)' }}>
                <AlertCircle size={11} /> Backend Offline
              </span>
            )}
            <span className="badge badge-indigo"><Zap size={11} /> Groq API</span>
          </div>
        </div>

        {/* Tab Content — all always mounted, CSS hides inactive ones */}
        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>

          <div className="page-body" style={{ display: activeTab === 'analysis' ? undefined : 'none' }}>
            <AnalysisTab
              file={file} setFile={setFile}
              stats={stats} setStats={setStats}
              profileHtml={profileHtml} setProfileHtml={setProfileHtml}
              analysisDone={analysisDone} setAnalysisDone={setAnalysisDone}
              featureSuggestions={featureSuggestions} setFeatureSuggestions={setFeatureSuggestions}
              onAnalysisComplete={handleAnalysisComplete}
            />
          </div>

          <div className="page-body" style={{ display: activeTab === 'compare' ? undefined : 'none' }}>
            <CompareTab
              hasDataset={!!stats}
              compareFile={compareFile} setCompareFile={setCompareFile}
              compareStats={compareStats} setCompareStats={setCompareStats}
            />
          </div>

          <div style={{ display: activeTab === 'chat' ? 'flex' : 'none', flex: 1, flexDirection: 'column', overflow: 'hidden' }}>
            <ChatTab
              messages={messages} setMessages={setMessages}
              initialAnalysis={initialAnalysis}
              availableModels={availableModels}
            />
          </div>

          <div className="page-body" style={{ display: activeTab === 'sandbox' ? undefined : 'none' }}>
            <SandboxTab hasDataset={!!stats} />
          </div>
        </div>
      </div>
    </div>
  );
}
