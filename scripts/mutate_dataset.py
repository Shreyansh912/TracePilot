"""
scripts/mutate_dataset.py
Generates 500+ synthetic bug-fix pairs via AST node mutations
and captures authentic Python runtime tracebacks using subprocess sandboxes.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Base template snippets representing canonical algorithmic patterns
BASE_TEMPLATES = [
    # 1. Division / Mathematical algorithms
    """def calculate_ratio(total, count):
    if count == 0:
        return 0.0
    return total / count""",

    """def compute_average(scores):
    if not scores:
        return 0.0
    return sum(scores) / len(scores)""",

    """def scale_dimensions(width, height, factor):
    if factor == 0:
        return width, height
    return width / factor, height / factor""",

    # 2. Sequence / Array indexing
    """def get_last_element(items):
    if not items:
        return None
    return items[-1]""",

    """def fetch_middle_item(elements):
    if not elements:
        return None
    mid = len(elements) // 2
    return elements[mid]""",

    """def pop_safely(lst):
    if len(lst) == 0:
        return None
    return lst.pop(0)""",

    # 3. Dictionary / Key lookup
    """def get_config_value(settings, key):
    if key not in settings:
        return None
    return settings[key]""",

    """def fetch_user_age(user_profile):
    return user_profile.get("age", 0)""",

    """def access_nested_data(payload):
    if "data" not in payload:
        return {}
    return payload["data"]""",

    # 4. Type Coercion / String concat
    """def format_user_id(prefix, num):
    return str(prefix) + str(num)""",

    """def join_status(code, message):
    return f"{code}: {message}" """
]


class BuggyMutator(ast.NodeTransformer):
    """
    Transforms AST nodes to introduce intentional runtime defects.
    """
    def __init__(self, mutation_type: str):
        self.mutation_type = mutation_type
        self.mutated = False

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Force Zero Division
        if self.mutation_type == "zero_div" and isinstance(node.op, ast.Div):
            self.mutated = True
            return ast.BinOp(left=node.left, op=ast.Div(), right=ast.Constant(value=0))

        # Break String concatenation with raw integers
        if self.mutation_type == "type_err" and isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Call) and getattr(node.right.func, "id", "") == "str":
                self.mutated = True
                return ast.BinOp(left=node.left, op=ast.Add(), right=ast.Constant(value=100))

        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Out of bounds index mutation
        if self.mutation_type == "index_oob":
            self.mutated = True
            return ast.Subscript(
                value=node.value,
                slice=ast.BinOp(left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[node.value], keywords=[]),
                                op=ast.Add(),
                                right=ast.Constant(value=5)),
                ctx=node.ctx
            )

        # Missing Key mutation
        if self.mutation_type == "key_err":
            self.mutated = True
            return ast.Subscript(
                value=node.value,
                slice=ast.Constant(value="__non_existent_key__"),
                ctx=node.ctx
            )

        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        # Strip protective guard clauses (e.g. `if count == 0: return 0.0`)
        if self.mutation_type == "strip_guard" and not self.mutated:
            self.mutated = True
            return ast.Pass()
        return self.generic_visit(node)


def generate_traceback(code_str: str) -> Optional[str]:
    """
    Executes Python code in an isolated subprocess to capture the actual traceback.
    """
    # Append driver invocation to trigger execution
    invocation_code = f"""
{code_str}

try:
    # Attempt common invocations
    if 'calculate_ratio' in locals(): calculate_ratio(10, 0)
    elif 'compute_average' in locals(): compute_average([])
    elif 'scale_dimensions' in locals(): scale_dimensions(10, 20, 0)
    elif 'get_last_element' in locals(): get_last_element([1, 2, 3])
    elif 'fetch_middle_item' in locals(): fetch_middle_item([1, 2])
    elif 'pop_safely' in locals(): pop_safely([])
    elif 'get_config_value' in locals(): get_config_value({{'name': 'test'}}, 'missing')
    elif 'fetch_user_age' in locals(): fetch_user_age({{'name': 'Alex'}})
    elif 'access_nested_data' in locals(): access_nested_data({{}})
    elif 'format_user_id' in locals(): format_user_id("ID_", 123)
    elif 'join_status' in locals(): join_status(200, "OK")
except Exception as e:
    import traceback
    traceback.print_exc()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(invocation_code)
        tmp_path = tmp.name

    try:
        res = subprocess.run(
            [sys.executable, "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if "Traceback (most recent call last):" in res.stderr:
            return res.stderr.strip()
    except Exception:
        pass
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return None


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "../datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_debugging_dataset_scaled.jsonl")

    mutation_modes = ["zero_div", "index_oob", "key_err", "type_err", "strip_guard"]
    records = []

    print("[*] Launching AST Bug Synthesizer...")

    iteration = 0
    target_records = 500

    while len(records) < target_records:
        for template in BASE_TEMPLATES:
            for mode in mutation_modes:
                if len(records) >= target_records:
                    break

                try:
                    tree = ast.parse(template)
                    mutator = BuggyMutator(mode)
                    mutated_tree = mutator.visit(tree)
                    ast.fix_missing_locations(mutated_tree)

                    buggy_code = ast.unparse(mutated_tree)
                    if buggy_code.strip() == template.strip():
                        continue

                    tb = generate_traceback(buggy_code)
                    if not tb:
                        continue

                    # Extract primary error line from traceback
                    err_lines = [l.strip() for l in tb.split("\n") if l.strip()]
                    err_summary = err_lines[-1] if err_lines else "RuntimeError"

                    record = {
                        "id": f"sample_{len(records):04d}",
                        "buggy_code": buggy_code,
                        "error_trace": tb,
                        "error_message": err_summary,
                        "fixed_code": template
                    }
                    records.append(record)

                    if len(records) % 50 == 0:
                        print(f"    -> Synthesized {len(records)} / {target_records} verifiable samples")
                except Exception:
                    continue

        iteration += 1
        if iteration > 50:
            break

    # Persist dataset
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[+] Successfully generated {len(records)} verified debug instances.")
    print(f"[+] Output saved to: {output_file}")


if __name__ == "__main__":
    main()