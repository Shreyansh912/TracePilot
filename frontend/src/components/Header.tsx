import React from 'react';
import { Terminal, Cpu, Database } from 'lucide-react';

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
      padding: '12px 24px',
      backgroundColor: '#161b22',
      borderBottom: '1px solid #30363d',
      color: '#f0f6fc'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Terminal color="#58a6ff" size={24} />
        <span style={{ fontSize: '1.2rem', fontWeight: 700, letterSpacing: '0.5px' }}>
          TRACEPILOT <span style={{ fontSize: '0.8rem', color: '#8b949e', fontWeight: 400 }}>| AI Debugging Assistant</span>
        </span>
      </div>

      <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={16} color="#7ee787" />
          <span style={{ color: '#8b949e' }}>Engine:</span>
          <span style={{ color: '#7ee787', fontWeight: 600 }}>{modelStatus}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Database size={16} color={dbStatus ? "#7ee787" : "#e3b341"} />
          <span style={{ color: '#8b949e' }}>Storage:</span>
          <span style={{ color: dbStatus ? "#7ee787" : "#e3b341", fontWeight: 600 }}>
            {dbStatus ? "MongoDB Connected" : "Local Memory"}
          </span>
        </div>
      </div>
    </header>
  );
};