import React from 'react';
import * as Diff from 'diff';

interface DiffViewerProps {
  originalCode: string;
  repairedCode: string;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({ originalCode, repairedCode }) => {
  const diffParts = Diff.diffLines(originalCode, repairedCode);

  return (
    <div style={{
      backgroundColor: '#0d1117',
      borderRadius: '6px',
      border: '1px solid #30363d',
      fontFamily: 'Consolas, Monaco, monospace',
      fontSize: '0.88rem',
      overflowX: 'auto',
      padding: '8px 0'
    }}>
      {diffParts.map((part, index) => {
        let bg = 'transparent';
        let color = '#c9d1d9';
        let prefix = '  ';

        if (part.added) {
          bg = 'rgba(46, 160, 67, 0.15)';
          color = '#7ee787';
          prefix = '+ ';
        } else if (part.removed) {
          bg = 'rgba(248, 81, 73, 0.15)';
          color = '#f85149';
          prefix = '- ';
        }

        return (
          <div
            key={index}
            style={{
              backgroundColor: bg,
              color: color,
              padding: '2px 12px',
              whiteSpace: 'pre'
            }}
          >
            {part.value.split('\n').filter((l, i, arr) => i < arr.length - 1 || l.length > 0).map((line, lIdx) => (
              <div key={lIdx}>{prefix}{line}</div>
            ))}
          </div>
        );
      })}
    </div>
  );
};