from pathlib import Path

from torch import nn
from torch_geometric.nn import GATv2Conv


class TensorLogger:
    def __init__(self, model, include_types=(nn.Linear, nn.Embedding, nn.LayerNorm, GATv2Conv)):
        self.model         = model
        self.include_types = include_types
        self.records       = []
        self.hooks         = []

    def _hook(self, name):
        def fn(module, inputs, output):
            if len(inputs) > 0:
                x        = inputs[0]
                in_shape = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            else:
                in_shape = "N/A"
            out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            self.records.append((name, module.__class__.__name__, in_shape, out_shape))
        return fn

    def attach(self):
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))
        return self

    def detach(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        self.records.clear()

    def _to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda r: r[0])

        col_names = ["Layer", "Type", "Input shape", "Output shape"]
        col_data  = [
            [f"`{r[0]}`" for r in rows],
            [str(r[1]) for r in rows],
            [str(r[2]) for r in rows],
            [str(r[3]) for r in rows],
        ]

        widths = []
        for header, data in zip(col_names, col_data):
            widths.append(max([len(header)] + [len(v) for v in data]) if rows else len(header))

        def fmt_row(cells):
            return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join((":" + "-" * (w - 1)) if w > 1 else "-" for w in widths) + " |"

        lines = [f"# {title}\n", fmt_row(col_names), fmt_sep()]
        for (name, typ, ins, outs), layer_txt in zip(rows, col_data[0]):
            lines.append(fmt_row([layer_txt, str(typ), str(ins), str(outs)]))

        lines.append(f"\n**Records:** {len(rows)}")
        return "\n".join(lines)

    def save_markdown(self, path, title: str = "Shape Log", sort_by_layer: bool = False):
        Path(path).write_text(self._to_markdown(title=title, sort_by_layer=sort_by_layer), encoding="utf-8")


class ModelSummary:
    def __init__(self, model: nn.Module):
        self.model        = model
        self.rows         = []
        self.total_params = 0

    def _count_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def run(self):
        self.total_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue

            n_params           = self._count_params(module)
            self.total_params += n_params
            self.rows.append((name, module.__class__.__name__, n_params))

    def _to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo layers found."

        rows_fmt = [(name, typ, f"{params:,}") for name, typ, params in self.rows]

        col1 = max(len("Layer"), *(len(name) for name, _, _ in rows_fmt))
        col2 = max(len("Type"), *(len(typ) for _, typ, _ in rows_fmt))
        col3 = max(len("Parameters"), *(len(p) for _, _, p in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = [line("Layer", "Type", "Parameters"), f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |"]
        for name, typ, params in rows_fmt:
            table.append(line(name, typ, params))

        md = [f"# {title}\n", *table, f"\n**Total Parameters:** `{self.total_params:,}`"]
        return "\n".join(md)

    def save_markdown(self, path: str, title: str = "Model Summary"):
        Path(path).write_text(self._to_markdown(title=title), encoding="utf-8")
