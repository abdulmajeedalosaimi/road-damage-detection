"""After editing m2m3_*.py, refresh the standalone notebook's embedded sources."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
path = root / "Road_Damage_M2_M3_Colab.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
sources = {p.name:p.read_text(encoding="utf-8-sig") for p in sorted(root.glob("m2m3_*.py"))}
for source in sources.values():
    compile(source, "<module>", "exec")
matches = 0
for cell in notebook["cells"]:
    source = cell["source"]
    if isinstance(source,list):
        source = "".join(source)
    if cell["cell_type"] == "code" and source.startswith("MODULE_SOURCES = "):
        _, rest = source.split("\n",1)
        cell["source"] = "MODULE_SOURCES = " + repr(sources) + "\n" + rest
        cell["execution_count"] = None
        cell["outputs"] = []
        matches += 1
if matches != 1:
    raise ValueError(f"Expected one embedded-source cell, found {matches}")
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]) if isinstance(cell["source"],list) else cell["source"],cell["id"],"exec")
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
print("Notebook sources updated. Use a new results folder for changed code.")

