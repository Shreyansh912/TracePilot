"""
backend/app/debugging/traceback_parser.py
Extracts file names, line numbers, function names, and error types
from Python exception tracebacks using regex pattern matching.
"""
import re
from typing import Optional, Dict, Any


class TracebackParser:
    # Pattern to match Python traceback frames: File "...", line X, in Y
    FRAME_REGEX = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+([^\n]+))?')
    ERROR_REGEX = re.compile(r'([a-zA-Z_]\w*(?:Error|Exception|Warning|Interrupt)):\s*(.*)')

    @classmethod
    def parse(cls, trace_text: str) -> Dict[str, Any]:
        """
        Parses a traceback string and returns the most proximate offending frame.
        """
        lines = trace_text.strip().splitlines()
        frames = []
        error_type = "RuntimeError"
        error_message = trace_text.strip()

        for line in lines:
            frame_match = cls.FRAME_REGEX.search(line)
            if frame_match:
                filename, lineno, funcname = frame_match.groups()
                frames.append({
                    "file": filename,
                    "line": int(lineno),
                    "function": funcname.strip() if funcname else "module"
                })

            error_match = cls.ERROR_REGEX.search(line)
            if error_match:
                error_type = error_match.group(1)
                error_message = error_match.group(2).strip()

        # The active error site is the last frame in the stack
        offending_frame = frames[-1] if frames else None

        return {
            "error_type": error_type,
            "error_message": error_message,
            "line": offending_frame["line"] if offending_frame else None,
            "function": offending_frame["function"] if offending_frame else None,
            "file": offending_frame["file"] if offending_frame else None,
            "frame_depth": len(frames)
        }