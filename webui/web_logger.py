from datetime import datetime


class WebLogger:
    COLORS = {
        "ok"      : "\033[32m",
        "muted"   : "\033[2m",
        "warning" : "\033[33m",
        "error"   : "\033[31m",
    }
    RESET = "\033[0m"

    def _emit(self, kind, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        color = self.COLORS[kind]
        print(f"{self.COLORS['muted']}{stamp}{self.RESET} {color}{message}{self.RESET}", flush=True)

    def ok(self, message):
        self._emit("ok", message)

    def muted(self, message):
        self._emit("muted", message)

    def warning(self, message):
        self._emit("warning", message)

    def error(self, message):
        self._emit("error", message)
