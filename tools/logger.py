import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


_THEME = Theme({
    "section":    "bold cyan",
    "subsection": "white",
    "key":        "bold magenta",
    "value":      "bright_white",
    "logging.level.debug":    "white",
    "logging.level.info":     "white",
    "logging.level.warning":  "bold yellow",
    "logging.level.error":    "bold red",
    "logging.level.critical": "bold red",
})

_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(theme=_THEME, highlight=False, soft_wrap=False)
    return _CONSOLE


class Logger:
    LOG_LEVELS = {
        'DEBUG'    : logging.DEBUG,
        'INFO'     : logging.INFO,
        'WARNING'  : logging.WARNING,
        'ERROR'    : logging.ERROR,
        'CRITICAL' : logging.CRITICAL,
    }

    def __init__(self, log_dir: Optional[str] = None, name: str = "experiment", level: str = "INFO"):
        self.log_dir    = log_dir
        self.name       = name
        self.start_time = datetime.now()

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.console = get_console()
        self.logger  = logging.getLogger(name)
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

        log_level = self.LOG_LEVELS.get(str(level).upper(), logging.INFO)
        self.logger.setLevel(log_level)

        rich_handler = RichHandler(
            console         = self.console,
            level           = log_level,
            show_time       = True,
            show_level      = False,
            show_path       = False,
            markup          = False,
            rich_tracebacks = True,
            log_time_format = "[%H:%M:%S]",
        )
        self.logger.addHandler(rich_handler)

        self._file_handler: Optional[logging.FileHandler] = None
        if log_dir:
            file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler   = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), mode='w', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
            self._file_handler = file_handler

    def section(self, title: str) -> None:
        text = str(title).upper()
        self.console.print()
        self.console.print(Rule(Text(text, style="section"), style="cyan"))
        self._to_file(f">>> {text}")

    def subsection(self, title: str) -> None:
        self.console.print(f"  [cyan]>[/cyan] {title}", style="bold white", markup=True)
        self._to_file(f"  > {title}")

    def kv_table(self, data: Mapping[str, Any], title: Optional[str] = None) -> None:
        table = Table(title=title, show_header=True, header_style="bold cyan", expand=False)
        table.add_column("Field", style="key", no_wrap=True)
        table.add_column("Value", style="value")
        for key, value in data.items():
            table.add_row(str(key), str(value))
        self.console.print(table)
        self._to_file(f"{title or 'table'}: " + ", ".join(f"{k}={v}" for k, v in data.items()))

    def _to_file(self, message: str) -> None:
        if self._file_handler is not None:
            self._file_handler.handle(self.logger.makeRecord(self.name, logging.INFO, "", 0, message, None, None))

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def save_profiler_results(self, stats, output: str) -> str:
        ordered   = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)
        col_names = ["Function", "Calls", "Total Time (s)", "Per Call (s)", "Cumulative Time (s)", "Cumulative Per Call (s)", "Location"]

        rows = []
        for func, (cc, nc, tt, ct, callers) in ordered:
            filename, lineno, func_name = func

            per_call_total = tt / nc if nc else 0.0
            per_call_cum   = ct / nc if nc else 0.0

            rows.append([func_name, str(nc), f"{tt:.6f}", f"{per_call_total:.6f}", f"{ct:.6f}", f"{per_call_cum:.6f}", f"{filename}:{lineno}"])

        widths = [max([len(col_names[index])] + [len(row[index]) for row in rows]) for index in range(len(col_names))]

        def fmt_row(cells):
            return "| " + " | ".join(f"{cell:<{width}}" for cell, width in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join("-" * width for width in widths) + " |"

        lines = [f"# Profiler Results\n", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", fmt_row(col_names), fmt_sep()]
        lines.extend(fmt_row(row) for row in rows)

        Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.info(f"Full profiler results saved to: {output}")

        return output

    def close(self) -> None:
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info(f"[End] Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
        self._file_handler = None


class NullLogger:
    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None
