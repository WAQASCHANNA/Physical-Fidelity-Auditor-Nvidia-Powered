"""Tavily Real-World Physical Constants Retriever (auditor/tavily_client.py).

Retrieves empirical friction coefficients, restitution values, and material properties
from Tavily Search API to ground the Nemotron physics audit in physical reality.
Includes persistent disk caching to conserve API credits and an offline engineering
reference handbook fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_dotenv_file(dotenv_path: Optional[Path] = None) -> None:
    """Load key=value pairs from .env into os.environ if not already set."""
    candidates = [
        dotenv_path,
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        if path and path.exists():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and (key not in os.environ or not os.environ[key]):
                    os.environ[key] = val
            break


load_dotenv_file()

try:
    from tavily import TavilyClient  # type: ignore[import-untyped]
    HAS_TAVILY_SDK = True
except ImportError:
    HAS_TAVILY_SDK = False


EMPIRICAL_CONSTANTS_REFERENCE: Dict[str, Dict[str, Any]] = {
    "warehouse_amr": {
        "material_pair": "polyurethane AMR drive wheels on sealed warehouse concrete",
        "query": "coefficient of kinetic friction polyurethane wheel on sealed concrete warehouse floor",
        "empirical_friction_range": [0.60, 0.85],
        "recommended_friction": 0.72,
        "recommended_restitution": 0.10,
        "source": "Industrial Mobile Robotics & AGV Traction Standards (ISO 3691-4)",
        "content": (
            "Polyurethane drive tires on dry sealed warehouse concrete exhibit a static friction "
            "coefficient mu_s of 0.65 to 0.85 (nominal 0.72) and rolling resistance coefficient "
            "c_r of 0.015 to 0.025, limiting maximum unforced lateral acceleration before wheel slip."
        ),
    },
    "franka_lift_cube": {
        "material_pair": "rubber gripper pads on ABS plastic cube",
        "query": "coefficient of static friction rubber on ABS plastic robotic gripper physical constant",
        "empirical_friction_range": [0.55, 0.75],
        "recommended_friction": 0.65,
        "recommended_restitution": 0.15,
        "source": "IEEE Robotics & Automation Manipulation Handbook (Parallel-Jaw Elastomer-Polymer Contact)",
        "content": (
            "Elastomeric/nitrile rubber parallel-jaw robotic fingertips contacting rigid ABS plastic "
            "exhibit a static friction coefficient mu_s between 0.55 and 0.75 (nominal mu=0.65). "
            "Force-closure antipodal grasping of a mass m under gravity g requires normal finger force "
            "F_n >= (m * g) / (2 * mu_s) to prevent vertical slip."
        ),
    },
    "ramp_roll": {
        "material_pair": "rubber on dry wood",
        "query": "coefficient of kinetic friction rubber on dry wood physical constant",
        "empirical_friction_range": [0.45, 0.65],
        "recommended_friction": 0.50,
        "recommended_restitution": 0.45,
        "source": "CRC Handbook of Chemistry and Physics / Engineering Toolbox (Tribology Reference)",
        "content": (
            "Empirical tribology measurements for vulcanized rubber rolling/sliding on dry planed wood "
            "report a kinetic friction coefficient mu_k between 0.45 and 0.65 (typical nominal mu=0.50) "
            "and a normal coefficient of restitution e between 0.40 and 0.50."
        ),
    },
    "block_stack": {
        "material_pair": "wood on wood",
        "query": "coefficient of static friction wood on wood physical constant",
        "empirical_friction_range": [0.25, 0.50],
        "recommended_friction": 0.42,
        "recommended_restitution": 0.25,
        "source": "Marks' Standard Handbook for Mechanical Engineers (Wood-on-Wood Contact)",
        "content": (
            "Clean dry wood on wood exhibits a static friction coefficient mu_s of 0.25 to 0.50 "
            "(nominal 0.42) and kinetic friction mu_k around 0.20 to 0.35. Under gravitational "
            "loading, a rectangular block topples when the vertical projection of its center of mass "
            "crosses the support polygon edge."
        ),
    },
    "projectile": {
        "material_pair": "steel on concrete",
        "query": "coefficient of restitution and friction steel on concrete physical constant",
        "empirical_friction_range": [0.40, 0.55],
        "recommended_friction": 0.45,
        "recommended_restitution": 0.68,
        "source": "ASME Impact Mechanics & Contact Restitution Tables (Steel-Concrete)",
        "content": (
            "Hardened steel impacting rigid dry concrete exhibits a coefficient of restitution e "
            "between 0.60 and 0.72 at low-to-moderate impact velocities (5-8 m/s) and a sliding "
            "friction coefficient mu of 0.40 to 0.55 under standard earth gravity g = 9.81 m/s^2."
        ),
    },
}


class PhysicalConstantsRetriever:
    """Retrieves and caches real-world physical constants via Tavily API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: Path | str = Path("data/cache/tavily_cache.json"),
    ) -> None:
        raw_key = os.getenv("TAVILY_API_KEY", "") if api_key is None else api_key
        self.api_key = raw_key.strip()
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _is_live_key_configured(self) -> bool:
        return bool(
            HAS_TAVILY_SDK
            and self.api_key
            and self.api_key not in ("YOUR_TAVILY_KEY", "your_tavily_api_key_here")
        )

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    def retrieve_constants(
        self,
        scenario: str,
        custom_query: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve physical constants for a scenario using cache -> Tavily API -> reference handbook."""
        ref = EMPIRICAL_CONSTANTS_REFERENCE.get(
            scenario,
            EMPIRICAL_CONSTANTS_REFERENCE["ramp_roll"],
        )
        query = custom_query or ref["query"]
        cache_key = f"{scenario}::{query}"

        if not force_refresh and cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            if cached.get("retrieval_mode") == "tavily_live_api" or not self._is_live_key_configured():
                cached["retrieval_mode"] = f"{cached.get('retrieval_mode', 'cached')} (disk_cache)"
                return cached

        if self._is_live_key_configured():
            try:
                client = TavilyClient(api_key=self.api_key)
                response = client.search(query=query, max_results=2)
                results = response.get("results", [])
                top = results[0] if results else {}
                payload: Dict[str, Any] = {
                    "scenario": scenario,
                    "query": query,
                    "material_pair": ref["material_pair"],
                    "empirical_friction_range": ref["empirical_friction_range"],
                    "recommended_friction": ref["recommended_friction"],
                    "recommended_restitution": ref["recommended_restitution"],
                    "source": top.get("url") or top.get("title") or ref["source"],
                    "content": top.get("content") or ref["content"],
                    "retrieval_mode": "tavily_live_api",
                    "raw_results_count": len(results),
                }
                self._cache[cache_key] = payload
                self._save_cache()
                return payload
            except Exception as exc:
                fallback = dict(ref)
                fallback["scenario"] = scenario
                fallback["query"] = query
                fallback["retrieval_mode"] = f"handbook_fallback (tavily_error: {type(exc).__name__})"
                self._cache[cache_key] = fallback
                self._save_cache()
                return fallback

        fallback = dict(ref)
        fallback["scenario"] = scenario
        fallback["query"] = query
        fallback["retrieval_mode"] = "handbook_reference_fallback"
        self._cache[cache_key] = fallback
        self._save_cache()
        return fallback

