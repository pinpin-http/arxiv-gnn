"""Lecture du journal de la sonde thermique de la machine de développement.

La machine qui a produit les résultats tourne avec une sonde qui, au-dessus
de 90 °C, abaisse temporairement le plafond CPU des calculs lourds et l'écrit
dans ~/.cache/heat-guard/temps.log (lignes « RALENTI », ou « PAUSE » pour une
ancienne version). Une mesure de temps qui chevauche un tel épisode est refaite.
Sans ce journal (autre machine, CI), aucune mesure n'est écartée.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

HEAT_LOG = Path(os.environ.get("ARXIV_GNN_HEAT_LOG", Path.home() / ".cache" / "heat-guard" / "temps.log"))

_EVENT = re.compile(r"^(?:(\d{4}-\d{2}-\d{2}) )?(\d{2}:\d{2}:\d{2}) (PAUSE|RALENTI)\b")
_STATE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \d+C .*(?:pause|ralenti)=1")


def throttle_events(start: float, end: float, path: Path = HEAT_LOG) -> list[str]:
    """Lignes du journal signalant un ralentissement entre `start` et `end`
    (secondes depuis l'epoch, heure locale), avec 5 s de marge."""
    if not path.exists():
        return []
    day = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
    hits = []
    for line in path.read_text(errors="replace").splitlines():
        m = _EVENT.match(line) or _STATE.match(line)
        if not m:
            continue
        stamp = datetime.strptime(f"{m.group(1) or day} {m.group(2)}", "%Y-%m-%d %H:%M:%S").timestamp()
        if start - 5 <= stamp <= end + 5:
            hits.append(line)
    return hits
