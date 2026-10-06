import React from 'react';
import { Terminal, Cpu, Database, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  modelStatus: string;
  dbStatus: boolean;
}

export const Header: React.FC<HeaderProps> = ({ modelStatus, dbStatus }) => {
  return (
    <header style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 28px',
      backgroundColor: '#0d1117',
      borderBottom: '1px solid #21262d',
      color: '#f0f6fc'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          backgroundColor: '#1f6feb',
          padding: '6px',
          borderRadius: '6px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Terminal size={18} color="#ffffff" />
        </div>
        <div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, letterSpacing: '0.5px' }}>
            TracePilot <span style={{ color: '#58a6ff', fontSize: '0.8rem', fontWeight: 500, marginLeft: '6px' }}>v0.1-alpha</span>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#8b949e' }}>
            Autonomous Python SLM Debugging & AST Repair Sandbox
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          backgroundColor: '#161b22',
          padding: '5px 12px',
          borderRadius: '20px',
          border: '1px solid #30363d',
          fontSize: '0.78rem'
        }}>
          <Cpu size={14} color="#7ee787" />
          <span style={{ color: '#8b949e' }}>SLM:</span>
          <span style={{ color: '#7ee787', fontWeight: 600 }}>{modelStatus}</span>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          backgroundColor: '#161b22',
          padding: '5px 12px',
          borderRadius: '20px',
          border: '1px solid #30363d',
          fontSize: '0.78rem'
        }}>
          <Database size={14} color={dbStatus ? "#7ee787" : "#d29922"} />
          <span style={{ color: '#8b949e' }}>Storage:</span>
          <span style={{ color: dbStatus ? "#7ee787" : "#d29922", fontWeight: 600 }}>
            {dbStatus ? "MongoDB Connected" : "Local Memory"}
          </span>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          backgroundColor: '#161b22',
          padding: '5px 12px',
          borderRadius: '20px',
          border: '1px solid #30363d',
          fontSize: '0.78rem'
        }}>
          <ShieldCheck size={14} color="#58a6ff" />
          <span style={{ color: '#58a6ff', fontWeight: 600 }}>Sandbox Enforced</span>
        </div>
      </div>
    </header>
  );
};