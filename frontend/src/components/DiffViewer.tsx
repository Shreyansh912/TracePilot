import React from 'react';
import * as Diff from 'diff';

interface DiffViewerProps {
  originalCode: string;
  repairedCode: string;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({ originalCode, repairedCode }) => {
  const diffParts = Diff.diffLines(originalCode.trim(), repairedCode.trim());

  return (
    <div style={{
      backgroundColor: '#0a0d12',
      borderRadius: '8px',
      border: '1px solid #30363d',
      fontFamily: '"JetBrains Mono", Consolas, "Courier New", monospace',
      fontSize: '0.85rem',
      lineHeight: '1.5',
      overflowX: 'auto',
      maxHeight: '380px',
      padding: '8px 0'
    }}>
      {diffParts.map((part, index) => {
        let bg = 'transparent';
        let color = '#adbac7';
        let prefix = ' ';

        if (part.added) {
          bg = 'rgba(46, 160, 67, 0.18)';
          color = '#7ee787';
          prefix = '+';
        } else if (part.removed) {
          bg = 'rgba(248, 81, 73, 0.18)';
          color = '#f85149';
          prefix = '-';
        }

        const lines = part.value.split('\n');
        if (lines[lines.length - 1] === '') lines.pop();

        return (
          <div key={index} style={{ backgroundColor: bg }}>
            {lines.map((line, lIdx) => (
              <div
                key={lIdx}
                style={{
                  display: 'flex',
                  padding: '1px 12px',
                  color: color,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}
              >
                <span style={{
                  width: '20px',
                  userSelect: 'none',
                  color: part.added ? '#7ee787' : part.removed ? '#f85149' : '#484f58',
                  fontWeight: 600
                }}>
                  {prefix}
                </span>
                <span style={{ flex: 1 }}>{line || ' '}</span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
};