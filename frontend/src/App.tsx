import React, { useState } from 'react';
import { Header } from './components/Header';
import { DiffViewer } from './components/DiffViewer';
import { Play, CheckCircle, AlertTriangle, RefreshCw, Code2, AlertCircle, Wrench, TerminalSquare, RotateCcw } from 'lucide-react';

interface IterationAttempt {
  attempt_number: number;
  patch_code: string;
  passed: boolean;
  error_summary: string;
}

interface DebugResult {
  error_type: string;
  error_location: string;
  root_cause: string;
  suggested_fix: string;
  syntax_valid: boolean;
  confidence_score: number;
  sandbox_executed?: boolean;
  sandbox_passed?: boolean;
  sandbox_stdout?: string;
  sandbox_stderr?: string;
  iterations_used?: number;
  self_corrected?: boolean;
  attempts?: IterationAttempt[];
}

const PRESETS = [
  {
    name: 'ZeroDivisionError',
    code: 'def calculate_ratio(total, count):\n    return total / count\n\nprint(calculate_ratio(10, 0))',
    trace: 'ZeroDivisionError: division by zero\n  File "main.py", line 2, in calculate_ratio'
  },
  {
    name: 'IndexError',
    code: 'def get_last_element(items):\n    return items[len(items) + 5]\n\nval = get_last_element([10, 20, 30])',
    trace: 'IndexError: list index out of range\n  File "main.py", line 2, in get_last_element'
  },
  {
    name: 'KeyError',
    code: 'def get_config_value(settings, key):\n    return settings["__non_existent_key__"]\n\ncfg = {"name": "app"}\nprint(get_config_value(cfg, "debug"))',
    trace: 'KeyError: \'__non_existent_key__\'\n  File "main.py", line 2, in get_config_value'
  }
];

export const App: React.FC = () => {
  const [code, setCode] = useState<string>(PRESETS[0].code);
  const [trace, setTrace] = useState<string>(PRESETS[0].trace);
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
        throw new Error(errorJson?.detail || `API returned status: ${response.status}`);
      }

      const data: DebugResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Connection failed to local backend service.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#010409',
      color: '#e6edf3',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
    }}>
      <Header modelStatus="MiniGPT-Debugger (4.9M)" dbStatus={true} />

      {/* Preset Selector */}
      <div style={{
        padding: '10px 28px',
        backgroundColor: '#0d1117',
        borderBottom: '1px solid #21262d',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <span style={{ fontSize: '0.78rem', color: '#8b949e', fontWeight: 600 }}>LOAD TEST SAMPLE:</span>
        {PRESETS.map((preset) => (
          <button
            key={preset.name}
            onClick={() => {
              setCode(preset.code);
              setTrace(preset.trace);
              setResult(null);
              setErrorMsg(null);
            }}
            style={{
              backgroundColor: '#161b22',
              border: '1px solid #30363d',
              borderRadius: '6px',
              color: '#58a6ff',
              padding: '4px 10px',
              fontSize: '0.78rem',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <main style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(420px, 1fr) minmax(460px, 1.2fr)',
        gap: '20px',
        padding: '20px 28px',
        flex: 1,
        boxSizing: 'border-box'
      }}>
        {/* Left Column: Code and Error Input */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 14px',
              backgroundColor: '#161b22',
              borderBottom: '1px solid #30363d',
              fontSize: '0.78rem',
              fontWeight: 600,
              color: '#c9d1d9'
            }}>
              <Code2 size={15} color="#58a6ff" />
              PYTHON SOURCE CODE
            </div>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              spellCheck={false}
              style={{
                width: '100%',
                height: '240px',
                backgroundColor: '#0d1117',
                border: 'none',
                padding: '14px',
                color: '#e6edf3',
                fontFamily: '"JetBrains Mono", Consolas, monospace',
                fontSize: '0.88rem',
                lineHeight: '1.45',
                resize: 'none',
                outline: 'none',
                boxSizing: 'border-box'
              }}
            />
          </div>

          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 14px',
              backgroundColor: '#161b22',
              borderBottom: '1px solid #30363d',
              fontSize: '0.78rem',
              fontWeight: 600,
              color: '#c9d1d9'
            }}>
              <AlertCircle size={15} color="#f85149" />
              STACK TRACE / DIAGNOSTIC
            </div>
            <textarea
              value={trace}
              onChange={(e) => setTrace(e.target.value)}
              spellCheck={false}
              style={{
                width: '100%',
                height: '110px',
                backgroundColor: '#0d1117',
                border: 'none',
                padding: '12px 14px',
                color: '#ff7b72',
                fontFamily: '"JetBrains Mono", Consolas, monospace',
                fontSize: '0.84rem',
                lineHeight: '1.4',
                resize: 'none',
                outline: 'none',
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
              gap: '10px',
              padding: '12px',
              backgroundColor: loading ? '#21262d' : '#238636',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              fontWeight: 600,
              fontSize: '0.92rem',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s'
            }}
          >
            {loading ? <RefreshCw className="animate-spin" size={18} /> : <Play size={18} />}
            {loading ? 'Synthesizing Patch via MiniGPT...' : 'Analyze & Synthesize Patch'}
          </button>

          {errorMsg && (
            <div style={{
              padding: '12px',
              backgroundColor: 'rgba(248, 81, 73, 0.12)',
              border: '1px solid #f85149',
              borderRadius: '6px',
              color: '#ff7b72',
              fontSize: '0.84rem'
            }}>
              <strong>Server Notice:</strong> {errorMsg}
            </div>
          )}
        </section>

        {/* Right Column: Inspection & Diff Viewer */}
        <section style={{
          backgroundColor: '#0d1117',
          border: '1px solid #30363d',
          borderRadius: '8px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '10px 16px',
            backgroundColor: '#161b22',
            borderBottom: '1px solid #30363d'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.84rem', fontWeight: 600 }}>
              <Wrench size={16} color="#7ee787" />
              INSPECTION & PATCH VERIFICATION
            </div>
            {result && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                fontSize: '0.75rem',
                color: result.syntax_valid ? '#7ee787' : '#f85149',
                backgroundColor: result.syntax_valid ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                padding: '3px 10px',
                borderRadius: '12px'
              }}>
                {result.syntax_valid ? <CheckCircle size={13} /> : <AlertTriangle size={13} />}
                {result.syntax_valid ? 'AST Validated' : 'Syntax Invalid'}
              </div>
            )}
          </div>

          <div style={{ padding: '18px', flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {!result && !loading && (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1,
                color: '#6e7681',
                fontSize: '0.88rem',
                textAlign: 'center',
                gap: '8px'
              }}>
                <Code2 size={36} color="#30363d" />
                <span>Submit source code to inspect AST scope and generate repairs.</span>
              </div>
            )}

            {result && (
              <>
                {/* Badges Bar */}
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <div style={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '0.78rem'
                  }}>
                    <span style={{ color: '#8b949e' }}>Exception: </span>
                    <strong style={{ color: '#58a6ff' }}>{result.error_type}</strong>
                  </div>

                  <div style={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '0.78rem'
                  }}>
                    <span style={{ color: '#8b949e' }}>Scope: </span>
                    <strong style={{ color: '#c9d1d9' }}>{result.error_location}</strong>
                  </div>

                  {/* Sandbox Telemetry Badge */}
                  {result.sandbox_executed && (
                    <div style={{
                      backgroundColor: result.sandbox_passed ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      border: `1px solid ${result.sandbox_passed ? '#2ea043' : '#f85149'}`,
                      color: result.sandbox_passed ? '#7ee787' : '#f85149',
                      padding: '4px 10px',
                      borderRadius: '6px',
                      fontSize: '0.78rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px'
                    }}>
                      {result.sandbox_passed ? <CheckCircle size={13} /> : <AlertTriangle size={13} />}
                      <span>{result.sandbox_passed ? 'Sandbox: Exit 0' : 'Sandbox: Execution Error'}</span>
                    </div>
                  )}

                  {/* Self-Correction Loop Badge */}
                  {result.iterations_used && result.iterations_used > 1 && (
                    <div style={{
                      backgroundColor: 'rgba(187, 128, 9, 0.15)',
                      border: '1px solid #d29922',
                      color: '#e3b341',
                      padding: '4px 10px',
                      borderRadius: '6px',
                      fontSize: '0.78rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px'
                    }}>
                      <RotateCcw size={13} />
                      <span>{result.self_corrected ? `Self-Corrected (Pass on Iteration ${result.iterations_used})` : `Multi-Pass (${result.iterations_used} attempts)`}</span>
                    </div>
                  )}

                  <div style={{
                    backgroundColor: '#161b22',
                    border: '1px solid #30363d',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '0.78rem'
                  }}>
                    <span style={{ color: '#8b949e' }}>Confidence: </span>
                    <strong style={{ color: '#7ee787' }}>{(result.confidence_score * 100).toFixed(0)}%</strong>
                  </div>
                </div>

                {/* Root Cause Card */}
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#8b949e', marginBottom: '6px' }}>
                    DIAGNOSED ROOT CAUSE
                  </div>
                  <div style={{
                    backgroundColor: '#161b22',
                    border: '1px solid #21262d',
                    borderRadius: '6px',
                    padding: '10px 14px',
                    fontSize: '0.84rem',
                    color: '#c9d1d9'
                  }}>
                    {result.root_cause}
                  </div>
                </div>

                {/* Diff Viewer */}
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#8b949e', marginBottom: '6px' }}>
                    PROPOSED PATCH DIFF
                  </div>
                  <DiffViewer originalCode={code} repairedCode={result.suggested_fix} />
                </div>

                {/* Sandbox Output Logs */}
                {result.sandbox_executed && (
                  <div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: '#8b949e',
                      marginBottom: '6px'
                    }}>
                      <TerminalSquare size={14} color="#8b949e" />
                      SANDBOX EXECUTION LOGS
                    </div>
                    <div style={{
                      backgroundColor: '#0a0d12',
                      border: '1px solid #21262d',
                      borderRadius: '6px',
                      padding: '10px 14px',
                      fontFamily: '"JetBrains Mono", Consolas, monospace',
                      fontSize: '0.8rem',
                      lineHeight: '1.4',
                      color: result.sandbox_passed ? '#7ee787' : '#ff7b72',
                      maxHeight: '120px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {result.sandbox_passed 
                        ? (result.sandbox_stdout || 'Process finished with exit code 0 (Execution Verified).')
                        : (result.sandbox_stderr || 'Process execution failed with non-zero exit code.')}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;