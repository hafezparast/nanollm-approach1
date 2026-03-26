#!/usr/bin/env python3
"""Phase 2: Generate an interactive HTML visualization of litellm's dependency graph.

Creates a D3.js force-directed graph showing:
- Modules as nodes (sized by lines of code)
- Import dependencies as edges
- Color-coded by submodule category
- Clickable nodes showing function lists
"""

import json
import os
from collections import defaultdict


def build_submodule_graph(modules: list[dict]) -> dict:
    """Aggregate modules into submodule-level nodes for cleaner visualization."""
    submodules = defaultdict(lambda: {
        "lines": 0, "functions": [], "classes": [], "files": [],
        "imports_from": set(), "imported_by": set(),
    })

    # Aggregate stats
    for m in modules:
        parts = m["module"].split(".")
        sub = parts[0] if len(parts) <= 1 else parts[0]
        # Use first two levels for grouping
        if len(parts) >= 2:
            sub = parts[0] + "." + parts[1] if parts[0] != parts[1] else parts[0]
        else:
            sub = parts[0]

        # Simplify: group by top-level subdir
        top = parts[0] if parts[0] not in ("__init__", "main") else "(core)"
        if parts[0] in ("__init__", "main", "_logging", "_redis", "_service_logger",
                        "_uuid", "_version", "_lazy_imports", "_lazy_imports_registry",
                        "constants", "budget_manager", "router", "scheduler",
                        "setup_wizard", "cost_calculator", "timeout"):
            top = "(core)"

        submodules[top]["lines"] += m.get("lines", 0)
        submodules[top]["files"].append(m["module"])

        for func in m.get("functions", []):
            if not func["name"].startswith("_") and not func["is_method"]:
                submodules[top]["functions"].append(func["name"])

        for cls in m.get("classes", []):
            submodules[top]["classes"].append(cls["name"])

        # Track imports between submodules
        for imp in m.get("imports", []):
            imp_mod = imp.get("module", "")
            if imp_mod and not imp_mod.startswith((".", "__future__", "typing")):
                imp_parts = imp_mod.split(".")
                imp_top = imp_parts[0]
                if imp_top in ("__init__", "main", "_logging", "constants"):
                    imp_top = "(core)"
                if imp_top != top:
                    submodules[top]["imports_from"].add(imp_top)

    # Convert sets to lists for JSON
    for sub in submodules.values():
        sub["imports_from"] = sorted(sub["imports_from"])
        sub["imported_by"] = sorted(sub["imported_by"])
        sub["function_count"] = len(sub["functions"])
        sub["class_count"] = len(sub["classes"])
        sub["file_count"] = len(sub["files"])
        # Keep only unique function names
        sub["functions"] = sorted(set(sub["functions"]))[:30]  # Cap for display
        sub["classes"] = sorted(set(sub["classes"]))[:20]

    # Build imported_by from imports_from
    for name, data in submodules.items():
        for dep in data["imports_from"]:
            if dep in submodules:
                submodules[dep]["imported_by"].append(name)

    return dict(submodules)


def generate_html(submodules: dict, output_path: str):
    """Generate interactive D3.js visualization."""

    # Prepare nodes and edges
    nodes = []
    edges = []
    node_ids = set()

    # Color categories
    categories = {
        "(core)": "#e74c3c",
        "llms": "#3498db",
        "proxy": "#2ecc71",
        "types": "#9b59b6",
        "litellm_core_utils": "#f39c12",
        "integrations": "#1abc9c",
        "router_strategy": "#e67e22",
        "router_utils": "#e67e22",
    }
    default_color = "#95a5a6"

    for name, data in sorted(submodules.items(), key=lambda x: -x[1]["lines"]):
        # Skip tiny modules
        if data["lines"] < 50:
            continue
        nodes.append({
            "id": name,
            "lines": data["lines"],
            "functions": data["functions"],
            "classes": data["classes"],
            "function_count": data["function_count"],
            "class_count": data["class_count"],
            "file_count": data["file_count"],
            "color": categories.get(name, default_color),
        })
        node_ids.add(name)

    for name, data in submodules.items():
        if name not in node_ids:
            continue
        for dep in data["imports_from"]:
            if dep in node_ids:
                edges.append({"source": name, "target": dep})

    graph_data = json.dumps({"nodes": nodes, "edges": edges})

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>litellm Dependency Graph — NanoLLM Approach 1</title>
<style>
  body {{ margin: 0; font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; }}
  svg {{ width: 100vw; height: 100vh; }}
  .node circle {{ stroke: #fff; stroke-width: 1.5; cursor: pointer; }}
  .node text {{ font-size: 10px; fill: #ccc; pointer-events: none; }}
  .link {{ stroke: #555; stroke-opacity: 0.4; fill: none; }}
  .link.highlighted {{ stroke: #ff6b6b; stroke-opacity: 1; stroke-width: 2; }}
  #info-panel {{
    position: fixed; top: 10px; right: 10px; width: 350px; max-height: 90vh;
    background: rgba(30,30,50,0.95); border: 1px solid #444; border-radius: 8px;
    padding: 15px; overflow-y: auto; font-size: 13px; display: none;
  }}
  #info-panel h2 {{ margin: 0 0 10px 0; color: #ff6b6b; font-size: 16px; }}
  #info-panel .stat {{ color: #888; margin: 2px 0; }}
  #info-panel .func-list {{ color: #7ec8e3; margin: 5px 0; }}
  #info-panel .cls-list {{ color: #c39bd3; margin: 5px 0; }}
  #title {{
    position: fixed; top: 10px; left: 10px; font-size: 18px; font-weight: bold;
    color: #ff6b6b; text-shadow: 0 0 10px rgba(255,107,107,0.3);
  }}
  #subtitle {{ position: fixed; top: 35px; left: 10px; font-size: 12px; color: #888; }}
  #legend {{
    position: fixed; bottom: 10px; left: 10px; font-size: 11px; color: #888;
  }}
</style>
</head>
<body>
<div id="title">litellm Dependency Graph</div>
<div id="subtitle">NanoLLM Approach 1 — Click nodes to explore. Node size = lines of code.</div>
<div id="info-panel"></div>
<div id="legend">
  <b>Legend:</b>
  <span style="color:#e74c3c">● Core</span>
  <span style="color:#3498db">● LLM Providers</span>
  <span style="color:#2ecc71">● Proxy</span>
  <span style="color:#9b59b6">● Types</span>
  <span style="color:#f39c12">● Utils</span>
  <span style="color:#1abc9c">● Integrations</span>
  <span style="color:#95a5a6">● Other</span>
</div>
<svg></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {graph_data};
const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("svg");
const g = svg.append("g");

// Zoom
svg.call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)));

const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.lines) / 3 + 10));

const link = g.append("g").selectAll("line")
  .data(data.edges).join("line").attr("class", "link");

const node = g.append("g").selectAll("g")
  .data(data.nodes).join("g").attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end", (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }})
  );

node.append("circle")
  .attr("r", d => Math.max(5, Math.sqrt(d.lines) / 3))
  .attr("fill", d => d.color)
  .attr("opacity", 0.8);

node.append("text")
  .attr("dx", d => Math.sqrt(d.lines) / 3 + 5)
  .attr("dy", 4)
  .text(d => d.id);

node.on("click", (e, d) => {{
  const panel = document.getElementById("info-panel");
  let html = `<h2>${{d.id}}</h2>`;
  html += `<div class="stat">${{d.lines.toLocaleString()}} lines · ${{d.function_count}} functions · ${{d.class_count}} classes · ${{d.file_count}} files</div>`;
  if (d.functions.length > 0) {{
    html += `<div class="func-list"><b>Functions:</b><br>${{d.functions.map(f => `  ${{f}}()`).join("<br>")}}</div>`;
  }}
  if (d.classes.length > 0) {{
    html += `<div class="cls-list"><b>Classes:</b><br>${{d.classes.join("<br>")}}</div>`;
  }}
  panel.innerHTML = html;
  panel.style.display = "block";

  // Highlight connected edges
  link.attr("class", l =>
    (l.source.id === d.id || l.target.id === d.id) ? "link highlighted" : "link"
  );
}});

simulation.on("tick", () => {{
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Wrote {output_path}")


def main():
    output_dir = "/Users/maysamhafezparast/vscode/nanollm-approach1/analysis/output"
    with open(os.path.join(output_dir, "modules.json")) as f:
        modules = json.load(f)

    submodules = build_submodule_graph(modules)

    # Save submodule data
    with open(os.path.join(output_dir, "submodules.json"), "w") as f:
        json.dump(submodules, f, indent=2, default=list)
    print(f"Wrote submodules.json ({len(submodules)} submodules)")

    # Generate HTML visualization
    generate_html(submodules, os.path.join(output_dir, "dep_graph.html"))


if __name__ == "__main__":
    main()
