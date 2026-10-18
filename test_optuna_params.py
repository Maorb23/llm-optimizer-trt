import ast, json
from pathlib import Path

paths = list(Path(".").rglob("optuna_inference_optimization_lab.ipynb"))
if not paths:
    raise SystemExit("Updated notebook not found on this VM")

path = paths[0]
source = "\n".join(
    "".join(cell.get("source", []))
    for cell in json.loads(path.read_text())["cells"]
    if cell.get("cell_type") == "code"
)
tree = ast.parse(source)
fn = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "serialize_engine_config"
)

class BenchmarkFailure(Exception):
    pass

namespace = {"BenchmarkFailure": BenchmarkFailure}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)

config = {
    "disable_radix_cache": False,
    "enable_mixed_chunk": False,
    "disable_overlap_schedule": False,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled",
}
print("Notebook:", path)
print("Serialized:", namespace["serialize_engine_config"]("sglang", 128, config))
