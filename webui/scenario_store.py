"""Saved Lab scenarios, one JSON file per name."""

import json
import re

from core.shared.state import Job, Vehicle
from web_errors import ApiError


class ScenarioStore:
    """Lists, reads, writes and deletes named scenarios.

    Attributes:
        directory : Directory holding `<name>.json` files.
    """

    NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")

    def __init__(self, directory):
        self.directory = directory

    def _path(self, name):
        if not self.NAME.match(name):
            raise ApiError(400, f"scenario name '{name}' must match [A-Za-z0-9_-]{{1,48}}")

        return self.directory / f"{name}.json"

    def _existing(self, name):
        path = self._path(name)
        if not path.exists():
            raise ApiError(404, f"unknown scenario '{name}'")

        return path

    def _validated(self, name, body):
        try:
            jobs     = [Job.from_dict(job).to_dict() for job in body["jobs"]]
            vehicles = [Vehicle.from_dict(vehicle).to_dict() for vehicle in body["vehicles"]]
            depot    = [float(body["depot"][0]), float(body["depot"][1])]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ApiError(400, f"malformed scenario: {error!r}")

        assignment = body.get("assignment")
        if assignment is not None:
            assignment = {str(int(vehicle_id)): [int(job_id) for job_id in job_ids] for vehicle_id, job_ids in assignment.items()}

        return {"name": name, "jobs": jobs, "vehicles": vehicles, "depot": depot, "assignment": assignment, "template": body.get("template")}

    def list(self):
        """Return the saved scenarios, newest first, as {"name", "mtime", "jobs", "vehicles"}."""
        if not self.directory.is_dir():
            return []

        entries = []
        for path in self.directory.glob("*.json"):
            scenario = json.loads(path.read_text())
            entries.append({"name": path.stem, "mtime": path.stat().st_mtime, "jobs": len(scenario["jobs"]), "vehicles": len(scenario["vehicles"])})

        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        return entries

    def get(self, name):
        """Return one scenario as {"name", "jobs", "vehicles", "depot", "assignment", "template"}; 404 when unknown."""
        return json.loads(self._existing(name).read_text())

    def put(self, name, body):
        """Create or replace a scenario from a request body; 400 when the name or the body is malformed."""
        path     = self._path(name)
        scenario = self._validated(name, body)

        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scenario))
        return {"ok": True}

    def delete(self, name):
        """Delete a scenario; 404 when unknown."""
        self._existing(name).unlink()
        return {"ok": True}
