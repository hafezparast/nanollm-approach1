#!/usr/bin/env python3
"""Phase 1: Map litellm's entire codebase using AST analysis.

Uses Python's `ast` module to parse every .py file in litellm and extract:
- All function/method definitions (name, args, decorators, line number)
- All class definitions (name, bases, methods)
- All imports (what each module imports from where)
- All function calls (who calls whom)

Outputs a JSON graph that can be visualized and queried.
"""

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_file(filepath: str, base_dir: str) -> dict:
    """Parse a single Python file and extract its symbols."""
    rel_path = os.path.relpath(filepath, base_dir)
    module_path = rel_path.replace("/", ".").replace(".py", "").replace(".__init__", "")

    try:
        with open(filepath, "r", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return {"module": module_path, "file": rel_path, "error": "SyntaxError"}

    result = {
        "module": module_path,
        "file": rel_path,
        "lines": source.count("\n") + 1,
        "functions": [],
        "classes": [],
        "imports": [],
        "calls": [],
    }

    for node in ast.walk(tree):
        # Top-level and nested function definitions
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "args": [a.arg for a in node.args.args if a.arg != "self"],
                "decorators": [],
                "is_method": False,
            }
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    func_info["decorators"].append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    func_info["decorators"].append(
                        f"{_get_attr_chain(dec)}"
                    )
            result["functions"].append(func_info)

        # Class definitions
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(_get_attr_chain(base))

            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "line": item.lineno,
                    })

            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "bases": bases,
                "methods": [m["name"] for m in methods],
                "method_count": len(methods),
            })

            # Mark functions that are methods
            method_names_at_lines = {(item.name, item.lineno) for item in node.body
                                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for func in result["functions"]:
                if (func["name"], func["line"]) in method_names_at_lines:
                    func["is_method"] = True

        # Import statements
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append({
                    "type": "import",
                    "module": alias.name,
                    "name": alias.asname or alias.name,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                result["imports"].append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                })

        # Function calls
        elif isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name:
                result["calls"].append({
                    "name": call_name,
                    "line": getattr(node, "lineno", 0),
                })

    return result


def _get_attr_chain(node) -> str:
    """Resolve chained attribute access: a.b.c → 'a.b.c'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _get_call_name(node: ast.Call) -> str | None:
    """Extract the name of a function call."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return _get_attr_chain(node.func)
    return None


def scan_directory(base_dir: str) -> list[dict]:
    """Scan all .py files in a directory tree."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__ and hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__"))]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                result = parse_file(filepath, base_dir)
                results.append(result)
    return results


def build_summary(modules: list[dict]) -> dict:
    """Build summary statistics from parsed modules."""
    total_lines = sum(m.get("lines", 0) for m in modules)
    total_functions = sum(len(m.get("functions", [])) for m in modules)
    total_classes = sum(len(m.get("classes", [])) for m in modules)
    total_imports = sum(len(m.get("imports", [])) for m in modules)
    total_calls = sum(len(m.get("calls", [])) for m in modules)

    # Find most-called functions
    call_counts = defaultdict(int)
    for m in modules:
        for call in m.get("calls", []):
            call_counts[call["name"]] += 1

    # Find external dependencies (non-litellm imports)
    external_deps = set()
    internal_imports = set()
    for m in modules:
        for imp in m.get("imports", []):
            mod = imp.get("module", "")
            if mod.startswith("litellm"):
                internal_imports.add(mod)
            elif mod and not mod.startswith("."):
                # Top-level package name
                top = mod.split(".")[0]
                if top not in {"__future__", "typing", "typing_extensions",
                              "collections", "dataclasses", "abc", "enum",
                              "os", "sys", "json", "re", "time", "datetime",
                              "pathlib", "functools", "contextlib", "copy",
                              "hashlib", "base64", "uuid", "io", "logging",
                              "traceback", "inspect", "importlib", "warnings",
                              "asyncio", "concurrent", "threading", "urllib",
                              "textwrap", "struct", "hmac", "secrets",
                              "tempfile", "shutil", "glob", "socket",
                              "ssl", "email", "mimetypes", "binascii"}:
                    external_deps.add(top)

    # Top-level public API (functions in __init__.py or main.py)
    public_functions = []
    for m in modules:
        if m["module"] in ("litellm", "litellm.main"):
            for func in m.get("functions", []):
                if not func["name"].startswith("_") and not func["is_method"]:
                    public_functions.append({
                        "name": func["name"],
                        "module": m["module"],
                        "is_async": func["is_async"],
                        "args": func["args"],
                    })

    # Submodule breakdown
    submodules = defaultdict(lambda: {"lines": 0, "functions": 0, "classes": 0, "files": 0})
    for m in modules:
        parts = m["module"].split(".")
        sub = parts[1] if len(parts) > 1 else "(root)"
        submodules[sub]["lines"] += m.get("lines", 0)
        submodules[sub]["functions"] += len(m.get("functions", []))
        submodules[sub]["classes"] += len(m.get("classes", []))
        submodules[sub]["files"] += 1

    top_called = sorted(call_counts.items(), key=lambda x: -x[1])[:50]

    return {
        "total_files": len(modules),
        "total_lines": total_lines,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "total_imports": total_imports,
        "total_call_sites": total_calls,
        "external_dependencies": sorted(external_deps),
        "external_dep_count": len(external_deps),
        "internal_import_targets": len(internal_imports),
        "public_api_functions": public_functions,
        "public_api_count": len(public_functions),
        "top_50_called_functions": [{"name": n, "count": c} for n, c in top_called],
        "submodules": dict(sorted(submodules.items(), key=lambda x: -x[1]["lines"])),
    }


def build_dependency_graph(modules: list[dict]) -> dict:
    """Build a module-level dependency graph from imports."""
    graph = {"nodes": [], "edges": []}
    module_set = {m["module"] for m in modules}

    for m in modules:
        graph["nodes"].append({
            "id": m["module"],
            "file": m.get("file", ""),
            "lines": m.get("lines", 0),
            "functions": len(m.get("functions", [])),
            "classes": len(m.get("classes", [])),
        })

        seen_edges = set()
        for imp in m.get("imports", []):
            target = imp.get("module", "")
            # Resolve relative imports
            if target.startswith("."):
                # Approximate: treat as litellm.X
                target = "litellm" + target

            # Only include edges to other litellm modules
            if target in module_set and target != m["module"]:
                edge_key = (m["module"], target)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    graph["edges"].append({
                        "source": m["module"],
                        "target": target,
                    })

    return graph


def build_function_call_graph(modules: list[dict]) -> dict:
    """Build a function-level call graph."""
    # Map function names to their defining module
    func_to_module = {}
    for m in modules:
        for func in m.get("functions", []):
            key = func["name"]
            if not func["is_method"]:
                func_to_module.setdefault(key, []).append(m["module"])

    # Build edges: module.caller → function_name
    edges = []
    for m in modules:
        for call in m.get("calls", []):
            call_name = call["name"]
            # Strip module prefix for simple names
            simple_name = call_name.split(".")[-1]
            if simple_name in func_to_module:
                for target_mod in func_to_module[simple_name]:
                    edges.append({
                        "caller_module": m["module"],
                        "callee": simple_name,
                        "callee_module": target_mod,
                        "line": call["line"],
                    })

    return {
        "total_edges": len(edges),
        "edges": edges[:1000],  # Cap to avoid huge output
        "function_locations": {k: v for k, v in func_to_module.items()
                               if not k.startswith("_") and len(v) == 1},
    }


def main():
    litellm_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/litellm_source/litellm"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/Users/maysamhafezparast/vscode/nanollm-approach1/analysis/output"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning {litellm_dir}...")
    modules = scan_directory(litellm_dir)
    print(f"Parsed {len(modules)} files")

    # Save raw module data
    with open(os.path.join(output_dir, "modules.json"), "w") as f:
        json.dump(modules, f, indent=2, default=str)
    print(f"Wrote modules.json")

    # Build and save summary
    summary = build_summary(modules)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary.json")

    # Build and save dependency graph
    dep_graph = build_dependency_graph(modules)
    with open(os.path.join(output_dir, "dep_graph.json"), "w") as f:
        json.dump(dep_graph, f, indent=2)
    print(f"Wrote dep_graph.json ({len(dep_graph['nodes'])} nodes, {len(dep_graph['edges'])} edges)")

    # Build and save function call graph
    call_graph = build_function_call_graph(modules)
    with open(os.path.join(output_dir, "call_graph.json"), "w") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Wrote call_graph.json ({call_graph['total_edges']} call edges)")

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"LITELLM CODEBASE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Files:              {summary['total_files']}")
    print(f"Lines of code:      {summary['total_lines']:,}")
    print(f"Functions/methods:  {summary['total_functions']:,}")
    print(f"Classes:            {summary['total_classes']:,}")
    print(f"Import statements:  {summary['total_imports']:,}")
    print(f"Call sites:         {summary['total_call_sites']:,}")
    print(f"External deps:      {summary['external_dep_count']} ({', '.join(summary['external_dependencies'][:15])}...)")
    print(f"Public API funcs:   {summary['public_api_count']}")
    print(f"\nTop 20 most-called functions:")
    for item in summary["top_50_called_functions"][:20]:
        print(f"  {item['count']:5d}x  {item['name']}")
    print(f"\nSubmodule breakdown (top 15 by lines):")
    for name, stats in list(summary["submodules"].items())[:15]:
        print(f"  {name:30s}  {stats['lines']:6d} lines  {stats['functions']:4d} funcs  {stats['classes']:3d} classes  {stats['files']:3d} files")
    print(f"\nPublic API functions ({summary['public_api_count']}):")
    for func in sorted(summary["public_api_functions"], key=lambda x: x["name"]):
        async_marker = "async " if func["is_async"] else ""
        print(f"  {async_marker}{func['name']}({', '.join(func['args'][:5])}{'...' if len(func['args']) > 5 else ''})")


if __name__ == "__main__":
    main()
