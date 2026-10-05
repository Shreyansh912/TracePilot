"""
backend/app/debugging/repair_engine.py
Coordinates the generate-verify-repair loop:
1. Synthesizes a patch via the local LLM.
2. Sandboxes execution to confirm the fix works.
3. If failure persists, feeds error back to model for up to MAX_REPAIRS.
"""
from typing import Dict, Any, Optional
from backend.app.sandbox.runner import CodeSandbox
from backend.app.debugging.context_builder import ContextBuilder


class RepairEngine:
    MAX_ATTEMPTS = 2

    def __init__(self, debug_generator):
        self.generator = debug_generator
        self.sandbox = CodeSandbox(timeout_seconds=2.5)

    def execute_and_verify(
        self,
        code: str,
        test_harness: Optional[str] = None
    ) -> Dict[str, Any]:
        target_code = f"{code}\n\n{test_harness}" if test_harness else code
        return self.sandbox.execute(target_code)

    def repair_loop(
        self,
        original_code: str,
        initial_error: str,
        test_harness: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes an iterative self-correction repair loop.
        """
        current_code = original_code
        current_error = initial_error
        attempts = 0
        repair_history = []

        while attempts < self.MAX_ATTEMPTS:
            attempts += 1
            # Step A: Build prompt
            prompt_data = ContextBuilder.assemble_prompt(
                code=current_code,
                error_message=current_error
            )

            # Step B: Generate patch from our local LLM
            generated_patch = self.generator.generate(
                prompt=prompt_data["prompt"],
                max_new_tokens=90,
                temperature=0.4,
                top_k=8,
                repetition_penalty=1.3
            ).strip()

            repair_history.append({
                "attempt": attempts,
                "proposed_patch": generated_patch
            })

            # Step C: Verify patch in isolated sandbox
            result = self.execute_and_verify(generated_patch, test_harness)

            if result["success"]:
                return {
                    "status": "VERIFIED_FIX",
                    "final_code": generated_patch,
                    "attempts": attempts,
                    "history": repair_history,
                    "verification": result
                }

            # Update state with the sandbox error for the next iteration
            current_code = generated_patch
            current_error = result["stderr"].strip() or "Syntax or Runtime Failure"

        return {
            "status": "UNVERIFIED_CANDIDATE",
            "final_code": generated_patch,
            "attempts": attempts,
            "history": repair_history,
            "verification": result
        }