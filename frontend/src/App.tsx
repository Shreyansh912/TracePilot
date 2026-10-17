import { useState, useEffect } from 'react';
import { 
  Play, 
  Code2, 
  Bug, 
  CheckCircle, 
  AlertTriangle, 
  Terminal, 
  History, 
  X, 
  Cpu, 
  ShieldCheck, 
  Database,
  ArrowRight,
  Clock
} from 'lucide-react';

interface Attempt {
  attempt_number: number;
  patch_code: string;
  passed: boolean;
  error_summary: string;
}

interface DebugResponse {
  error_type: string;
  error_location: string;
  root_cause: string;
  suggested_fix: string;
  syntax_valid: boolean;
  confidence_score: number;
  sandbox_executed: boolean;
  sandbox_passed: boolean;
  sandbox_stdout: string;
  sandbox_stderr: string;
  iterations_used: number;
  self_corrected: boolean;
  attempts: Attempt[];
}

interface HistoryItem {
  session_id: string;
  timestamp: string;
  error_type: string;
  sandbox_passed: boolean;
  confidence_score: number;
  code: string;
  stack_trace: string;
  suggested_fix: string;
  sandbox_stdout: string;
  sandbox_stderr: string;
}

const PRESETS = {
  ZeroDivisionError: {
    code: `def calculate_ratio(total, count):\n    return total / count\n\nprint(calculate_ratio(10, 0))`,
    trace: `ZeroDivisionError: division by zero\n File "main.py", line 2, in calculate_ratio`
  },
  IndexError: {
    code: `def get_first_element(items):\n    return items[0]\n\nprint(get_first_element([]))`,
    trace: `IndexError: list index out of range\n File "main.py", line 2, in get_first_element`
  },
  KeyError: {
    code: `def fetch_user_role(profile):\n    return profile['role']\n\nprint(fetch_user_role({'name': 'Alice'}))`,
    trace: `KeyError: 'role'\n File "main.py", line 2, in fetch_user_role`
  }
};

const API_BASE = 'http://127.0.0.1:8000';

function InlineDiffViewer({ original, modified }: { original: string; modified: string }) {
  const origLines = original.split('\n');
  const modLines = modified.split('\n');

  const origSet = new Set(origLines.map(l => l.trim()));
  const modSet = new Set(modLines.map(l => l.trim()));

  return (
    <div style={{
      backgroundColor: '#0d1117',
      fontFamily: 'JetBrains Mono, Menlo, Consolas, Monaco, monospace',
      fontSize: '0.8rem',
      padding: '12px',
      overflowX: 'auto',
      lineHeight: 1.5
    }}>
      {origLines.map((line, idx) => {
        const isRemoved = line.trim().length > 0 && !modSet.has(line.trim());
        if (!isRemoved) return null;
        return (
          <div key={`orig-${idx}`} style={{ backgroundColor: 'rgba(248, 81, 73, 0.15)', color: '#f85149', padding: '2px 6px', display: 'flex', gap: '8px' }}>
            <span style={{ width: '15px', userSelect: 'none' }}>-</span>
            <span>{line}</span>
          </div>
        );
      })}
      {modLines.map((line, idx) => {
        const isAdded = line.trim().length > 0 && !origSet.has(line.trim());
        return (
          <div
            key={`mod-${idx}`}
            style={{
              backgroundColor: isAdded ? 'rgba(46, 160, 67, 0.2)' : 'transparent',
              color: isAdded ? '#3fb950' : '#c9d1d9',
              padding: '2px 6px',
              display: 'flex',
              gap: '8px'
            }}
          >
            <span style={{ width: '15px', userSelect: 'none', color: isAdded ? '#3fb950' : '#484f58' }}>
              {isAdded ? '+' : ' '}
            </span>
            <span>{line}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [code, setCode] = useState(PRESETS.ZeroDivisionError.code);
  const [stackTrace, setStackTrace] = useState(PRESETS.ZeroDivisionError.trace);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DebugResponse | null>(null);
  
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [dbConnected, setDbConnected] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        // Support all standard MongoDB health status structures
        const isConnected = 
          data?.storage?.status === 'connected' ||
          data?.storage?.connected === true ||
          data?.database === 'connected' ||
          data?.mongodb === 'connected' ||
          data?.storage?.database_available === true ||
          data?.status === 'healthy';
        setDbConnected(Boolean(isConnected));
      } else {
        setDbConnected(false);
      }
    } catch {
      setDbConnected(false);
    }
  };

  const loadPreset = (key: keyof typeof PRESETS) => {
    setCode(PRESETS[key].code);
    setStackTrace(PRESETS[key].trace);
    setResult(null);
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setResult(null);
    try {
      const payload = {
        code,
        stack_trace: stackTrace,
        error_message: stackTrace.split('\n')[0]
      };
      const res = await fetch(`${API_BASE}/debug`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${res.status}`);
      }
      const data: DebugResponse = await res.json();
      setResult(data);
      checkHealth();
    } catch (err: any) {
      console.error(err);
      alert(err.message || 'Analysis request failed. Verify that your backend server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_BASE}/debug/history?limit=25`);
      if (!res.ok) throw new Error('Failed to fetch history');
      const data: HistoryItem[] = await res.json();
      setHistoryItems(data);
      setDbConnected(true);
    } catch (err) {
      console.error('Failed to load session history', err);
      setDbConnected(false);
    } finally {
      setHistoryLoading(false);
    }
  };

  const toggleHistoryDrawer = () => {
    if (!drawerOpen) {
      fetchHistory();
    }
    setDrawerOpen(!drawerOpen);
  };

  const restoreSession = (item: HistoryItem) => {
    setCode(item.code);
    setStackTrace(item.stack_trace);
    setResult({
      error_type: item.error_type,
      error_location: 'Restored from Session',
      root_cause: item.error_type,
      suggested_fix: item.suggested_fix,
      syntax_valid: true,
      confidence_score: item.confidence_score,
      sandbox_executed: true,
      sandbox_passed: item.sandbox_passed,
      sandbox_stdout: item.sandbox_stdout,
      sandbox_stderr: item.sandbox_stderr,
      iterations_used: 1,
      self_corrected: false,
      attempts: []
    });
    setDrawerOpen(false);
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0d1117', color: '#c9d1d9', fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      
      {/* Top Navigation Bar */}
      <header style={{
        height: '60px',
        borderBottom: '1px solid #30363d',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        backgroundColor: '#161b22'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #238636 0%, #2ea043 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontWeight: 700,
            fontSize: '1.1rem'
          }}>
            T
          </div>
          <div>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f0f6fc', letterSpacing: '0.3px' }}>TracePilot</div>
            <div style={{ fontSize: '0.72rem', color: '#8b949e' }}>Autonomous Code Repair Engine</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Storage Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.75rem',
            padding: '4px 10px',
            borderRadius: '20px',
            backgroundColor: dbConnected ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
            border: `1px solid ${dbConnected ? 'rgba(46, 160, 67, 0.4)' : 'rgba(248, 81, 73, 0.4)'}`,
            color: dbConnected ? '#3fb950' : '#f85149'
          }}>
            <Database size={13} />
            <span>{dbConnected ? 'MongoDB Connected' : 'MongoDB Disconnected'}</span>
          </div>

          {/* Sandbox Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.75rem',
            padding: '4px 10px',
            borderRadius: '20px',
            backgroundColor: 'rgba(56, 139, 253, 0.12)',
            border: '1px solid rgba(56, 139, 253, 0.35)',
            color: '#58a6ff'
          }}>
            <ShieldCheck size={13} />
            <span>Subprocess Sandbox</span>
          </div>

          {/* Session History Button */}
          <button
            onClick={toggleHistoryDrawer}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: '#21262d',
              color: '#c9d1d9',
              border: '1px solid #30363d',
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <History size={15} />
            <span>Session History</span>
          </button>
        </div>
      </header>

      {/* Preset Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '10px 24px',
        backgroundColor: '#0d1117',
        borderBottom: '1px solid #21262d',
        fontSize: '0.8rem'
      }}>
        <span style={{ color: '#8b949e', fontWeight: 600 }}>LOAD TEST SAMPLE:</span>
        {(['ZeroDivisionError', 'IndexError', 'KeyError'] as const).map((key) => (
          <button
            key={key}
            onClick={() => loadPreset(key)}
            style={{
              backgroundColor: '#21262d',
              color: '#58a6ff',
              border: '1px solid #30363d',
              borderRadius: '6px',
              padding: '4px 10px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 500
            }}
          >
            {key}
          </button>
        ))}
      </div>

      {/* Main Workspace Layout */}
      <main style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
        padding: '24px',
        maxWidth: '1600px',
        margin: '0 auto'
      }}>
        
        {/* Left Column: Code and Stack Trace Input */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Source Code Container */}
          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 14px',
              backgroundColor: '#161b22',
              borderBottom: '1px solid #30363d',
              fontSize: '0.78rem',
              fontWeight: 600,
              color: '#c9d1d9'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Code2 size={15} color="#58a6ff" />
                <span>PYTHON SOURCE CODE</span>
              </div>
              {result && result.sandbox_passed && (
                <button
                  onClick={() => setCode(result.suggested_fix)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    backgroundColor: '#238636',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '3px 9px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                  title="Replace editor code with verified fix"
                >
                  <CheckCircle size={12} />
                  Apply Fix
                </button>
              )}
            </div>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              rows={14}
              style={{
                width: '100%',
                padding: '14px',
                backgroundColor: '#0d1117',
                color: '#f0f6fc',
                fontFamily: 'JetBrains Mono, Menlo, Consolas, Monaco, monospace',
                fontSize: '0.85rem',
                border: 'none',
                outline: 'none',
                resize: 'vertical',
                lineHeight: 1.5,
                boxSizing: 'border-box'
              }}
            />
          </div>

          {/* Stack Trace Container */}
          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden'
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
              color: '#f85149'
            }}>
              <Bug size={15} color="#f85149" />
              <span>STACK TRACE / DIAGNOSTIC</span>
            </div>
            <textarea
              value={stackTrace}
              onChange={(e) => setStackTrace(e.target.value)}
              rows={5}
              style={{
                width: '100%',
                padding: '14px',
                backgroundColor: '#0d1117',
                color: '#f85149',
                fontFamily: 'JetBrains Mono, Menlo, Consolas, Monaco, monospace',
                fontSize: '0.8rem',
                border: 'none',
                outline: 'none',
                resize: 'vertical',
                lineHeight: 1.4,
                boxSizing: 'border-box'
              }}
            />
          </div>

          {/* Action Button */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              backgroundColor: loading ? '#21262d' : '#238636',
              color: '#ffffff',
              border: 'none',
              padding: '12px 20px',
              borderRadius: '6px',
              fontWeight: 600,
              fontSize: '0.95rem',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.15s ease'
            }}
          >
            {loading ? (
              <>
                <Cpu size={18} className="animate-spin" />
                Synthesizing Autonomous Patch...
              </>
            ) : (
              <>
                <Play size={18} fill="#ffffff" />
                Analyze & Synthesize Patch
              </>
            )}
          </button>
        </div>

        {/* Right Column: Inspection, Diff Viewer, & Sandbox Output */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Inspection & Patch Status Card */}
          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 14px',
              backgroundColor: '#161b22',
              borderBottom: '1px solid #30363d',
              fontSize: '0.78rem',
              fontWeight: 600,
              color: '#c9d1d9'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={15} color="#3fb950" />
                <span>INSPECTION & PATCH VERIFICATION</span>
              </div>
              {result && (
                <div style={{
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  backgroundColor: result.syntax_valid ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                  border: `1px solid ${result.syntax_valid ? 'rgba(46, 160, 67, 0.4)' : 'rgba(248, 81, 73, 0.4)'}`,
                  color: result.syntax_valid ? '#3fb950' : '#f85149'
                }}>
                  {result.syntax_valid ? '✓ AST Validated' : '⚠ Syntax Invalid'}
                </div>
              )}
            </div>

            <div style={{ padding: '14px' }}>
              {result ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  
                  {/* Status Pills */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    <span style={{ backgroundColor: '#21262d', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem' }}>
                      Exception: <strong style={{ color: '#58a6ff' }}>{result.error_type}</strong>
                    </span>
                    <span style={{ backgroundColor: '#21262d', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem' }}>
                      Scope: <strong style={{ color: '#f0f6fc' }}>{result.error_location}</strong>
                    </span>
                    <span style={{
                      backgroundColor: result.sandbox_passed ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      color: result.sandbox_passed ? '#3fb950' : '#f85149',
                      border: `1px solid ${result.sandbox_passed ? 'rgba(46, 160, 67, 0.3)' : 'rgba(248, 81, 73, 0.3)'}`,
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      fontWeight: 600
                    }}>
                      {result.sandbox_passed ? '✓ Sandbox: Exit 0' : '⚠ Sandbox: Execution Error'}
                    </span>
                    {result.iterations_used > 1 && (
                      <span style={{ backgroundColor: 'rgba(210, 153, 34, 0.15)', color: '#d29922', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', border: '1px solid rgba(210, 153, 34, 0.3)' }}>
                        ↻ Multi-Pass ({result.iterations_used} attempts)
                      </span>
                    )}
                    <span style={{ backgroundColor: '#21262d', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem' }}>
                      Confidence: <strong style={{ color: '#3fb950' }}>{Math.round(result.confidence_score * 100)}%</strong>
                    </span>
                  </div>

                  {/* Root Cause description */}
                  <div>
                    <div style={{ fontSize: '0.72rem', color: '#8b949e', fontWeight: 600, marginBottom: '4px', textTransform: 'uppercase' }}>
                      Diagnosed Root Cause
                    </div>
                    <div style={{ backgroundColor: '#161b22', padding: '8px 12px', borderRadius: '4px', fontSize: '0.8rem', color: '#c9d1d9', border: '1px solid #21262d' }}>
                      {result.root_cause}
                    </div>
                  </div>

                  {/* Patch Diff Section */}
                  <div>
                    <div style={{ fontSize: '0.72rem', color: '#8b949e', fontWeight: 600, marginBottom: '4px', textTransform: 'uppercase' }}>
                      Proposed Patch Diff
                    </div>
                    <div style={{ borderRadius: '6px', overflow: 'hidden', border: '1px solid #30363d' }}>
                      <InlineDiffViewer original={code} modified={result.suggested_fix} />
                    </div>
                  </div>

                </div>
              ) : (
                <div style={{ padding: '40px 0', textAlign: 'center', color: '#8b949e', fontSize: '0.85rem' }}>
                  Enter code and stack trace, then click "Analyze & Synthesize Patch" to trigger autonomous execution.
                </div>
              )}
            </div>
          </div>

          {/* Sandbox Execution Terminal Container */}
          <div style={{
            backgroundColor: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            overflow: 'hidden'
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
              <Terminal size={15} color="#58a6ff" />
              <span>SANDBOX EXECUTION LOGS</span>
            </div>
            <pre style={{
              margin: 0,
              padding: '14px',
              backgroundColor: '#010409',
              color: result?.sandbox_passed ? '#3fb950' : (result ? '#f85149' : '#8b949e'),
              fontFamily: 'JetBrains Mono, Menlo, Consolas, Monaco, monospace',
              fontSize: '0.8rem',
              minHeight: '80px',
              maxHeight: '160px',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap',
              lineHeight: 1.4
            }}>
              {result ? (
                result.sandbox_passed ? (
                  result.sandbox_stdout || 'Process finished with exit code 0 (Execution Verified).'
                ) : (
                  result.sandbox_stderr || 'Runtime error encountered during sandbox run.'
                )
              ) : (
                'Sandbox idle. Execution logs will render here.'
              )}
            </pre>
          </div>

        </div>

      </main>

      {/* Slide-out Session History Drawer */}
      {drawerOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          right: 0,
          bottom: 0,
          width: '380px',
          backgroundColor: '#161b22',
          borderLeft: '1px solid #30363d',
          boxShadow: '-8px 0 24px rgba(0,0,0,0.5)',
          zIndex: 100,
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Drawer Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 20px',
            borderBottom: '1px solid #30363d'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, fontSize: '0.95rem', color: '#f0f6fc' }}>
              <History size={18} color="#58a6ff" />
              Persisted Sessions
            </div>
            <button
              onClick={() => setDrawerOpen(false)}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: '#8b949e',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Drawer Content / List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {historyLoading ? (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: '0.85rem' }}>
                Loading session ledger...
              </div>
            ) : historyItems.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: '0.85rem' }}>
                No repair sessions found. Run a patch to record history.
              </div>
            ) : (
              historyItems.map((item) => (
                <div
                  key={item.session_id}
                  onClick={() => restoreSession(item)}
                  style={{
                    backgroundColor: '#0d1117',
                    border: '1px solid #30363d',
                    borderRadius: '6px',
                    padding: '12px',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s ease'
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#58a6ff')}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#30363d')}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: item.sandbox_passed ? '#3fb950' : '#f85149',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      {item.sandbox_passed ? <CheckCircle size={12} /> : <AlertTriangle size={12} />}
                      {item.error_type || 'Unknown Exception'}
                    </span>
                    <span style={{
                      fontSize: '0.68rem',
                      padding: '2px 6px',
                      borderRadius: '10px',
                      backgroundColor: item.sandbox_passed ? 'rgba(46, 160, 67, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      color: item.sandbox_passed ? '#3fb950' : '#f85149'
                    }}>
                      {item.sandbox_passed ? 'Passed' : 'Failed'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.7rem', color: '#8b949e', marginBottom: '8px' }}>
                    <Clock size={11} />
                    <span>{new Date(item.timestamp).toLocaleString()}</span>
                  </div>

                  <div style={{
                    fontSize: '0.75rem',
                    color: '#8b949e',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderTop: '1px solid #21262d',
                    paddingTop: '6px'
                  }}>
                    <span>Confidence: {Math.round(item.confidence_score * 100)}%</span>
                    <span style={{ color: '#58a6ff', display: 'flex', alignItems: 'center', gap: '2px' }}>
                      Replay <ArrowRight size={11} />
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

    </div>
  );
}