import React, { useState } from 'react';
import { Header } from './components/Header';
import { DiffViewer } from './components/DiffViewer';
import { Play, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';

interface DebugResult {
  error_type: string;
  error_location: string;
  root_cause: string;
  suggested_fix: string;
  syntax_valid: boolean;
  confidence_score: number;
}

export const App: React.FC = () => {
  const [code, setCode] = useState<string>(
    'def divide_numbers(a, b):\n    return a / b\n\nprint(divide_numbers(10, 0))'
  );
  const [trace, setTrace] = useState<string>(
    'ZeroDivisionError: division by zero\n  File "main.py", line 2, in divide_numbers'
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DebugResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          language: 'python',
          error_message: trace.split('\n')[0],
          stack_trace: trace
        })
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => null);
        throw new Error(errorJson?.detail || `API returned status ${response.status}`);
      }

      const data: DebugResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to communicate with TracePilot API.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0d1117', color: '#c9d1d9', display: 'flex', flexDirection: 'column' }}>
      <Header modelStatus="MiniGPT-Debugger (4.9M)" dbStatus={true} />

      <main style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', padding: '24px', flex: 1 }}>
        {/* Left Panel: Code & Error Inputs */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: '#8b949e' }}>
              SOURCE CODE
            </label>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              style={{
                width: '100%',
                height: '240px',
                backgroundColor: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '6px',
                padding: '12px',
                color: '#f0f6fc',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.9rem',
                resize: 'none',
                boxSizing: 'border-box'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: '#8b949e' }}>
              ERROR MESSAGE / STACK TRACE
            </label>
            <textarea
              value={trace}
              onChange={(e) => setTrace(e.target.value)}
              style={{
                width: '100%',
                height: '110px',
                backgroundColor: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '6px',
                padding: '12px',
                color: '#f85149',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.85rem',
                resize: 'none',
                boxSizing: 'border-box'
              }}
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '12px 20px',
              backgroundColor: loading ? '#21262d' : '#238636',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              fontWeight: 600,
              fontSize: '0.95rem',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? <RefreshCw className="animate-spin" size={18} /> : <Play size={18} />}
            {loading ? 'Analyzing Code & Running Transformer...' : 'Analyze & Repair Bug'}
          </button>

          {errorMsg && (
            <div style={{
              padding: '12px',
              backgroundColor: 'rgba(248, 81, 73, 0.1)',
              border: '1px solid #f85149',
              borderRadius: '6px',
              color: '#f85149',
              fontSize: '0.85rem',
              wordBreak: 'break-word'
            }}>
              <strong>Error:</strong> {errorMsg}
            </div>
          )}
        </section>

        {/* Right Panel: Diagnosis & Diff */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{
            backgroundColor: '#161b22',
            border: '1px solid #30363d',
            borderRadius: '6px',
            padding: '16px',
            minHeight: '400px'
          }}>
            <h2 style={{ fontSize: '1rem', fontWeight: 600, margin: '0 0 16px 0', color: '#f0f6fc' }}>
              DIAGNOSIS & VERIFIED PATCH
            </h2>

            {!result && !loading && (
              <div style={{ color: '#8b949e', fontSize: '0.9rem', textAlign: 'center', marginTop: '120px' }}>
                Run analysis to inspect bug location, root cause, and synthesized patch.
              </div>
            )}

            {result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  <span style={{ backgroundColor: '#21262d', padding: '4px 10px', borderRadius: '20px', fontSize: '0.8rem', color: '#58a6ff' }}>
                    Type: <strong>{result.error_type}</strong>
                  </span>
                  <span style={{ backgroundColor: '#21262d', padding: '4px 10px', borderRadius: '20px', fontSize: '0.8rem', color: '#8b949e' }}>
                    Location: <strong>{result.error_location}</strong>
                  </span>
                  <span style={{
                    backgroundColor: result.syntax_valid ? 'rgba(46, 160, 67, 0.2)' : 'rgba(248, 81, 73, 0.2)',
                    color: result.syntax_valid ? '#7ee787' : '#f85149',
                    padding: '4px 10px',
                    borderRadius: '20px',
                    fontSize: '0.8rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    {result.syntax_valid ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                    {result.syntax_valid ? 'Syntax Validated' : 'Syntax Invalid'}
                  </span>
                  <span style={{ backgroundColor: '#21262d', padding: '4px 10px', borderRadius: '20px', fontSize: '0.8rem', color: '#8b949e' }}>
                    Confidence: <strong>{(result.confidence_score * 100).toFixed(0)}%</strong>
                  </span>
                </div>

                <div>
                  <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: '4px' }}>ROOT CAUSE</div>
                  <div style={{ color: '#c9d1d9', fontSize: '0.9rem' }}>{result.root_cause}</div>
                </div>

                <div>
                  <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: '6px' }}>PATCH DIFF</div>
                  <DiffViewer originalCode={code} repairedCode={result.suggested_fix} />
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;