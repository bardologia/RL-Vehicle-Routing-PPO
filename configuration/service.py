from dataclasses import dataclass, field
from typing      import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter


@dataclass
class ServiceConfig:
    vroom_url: str = "http://localhost:3000"
    osrm_url: str = "http://localhost:5000"
    options: Dict[str, Any] = field(default_factory=lambda: {"g": True, "geometry": True, "threads": 8})
    _http_session: Optional[requests.Session] = field(init=False, default=None)

    @property
    def http_session(self) -> requests.Session:
        if self._http_session is None:
            sess = requests.Session()
            sess.headers.update({"Content-Type": "application/json"})
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._http_session = sess
        return self._http_session
