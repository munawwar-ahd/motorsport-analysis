#!/usr/bin/env python3
"""
PIT WALL data pipeline.

Reads the Ergast-style CSV archive in this folder and writes compact JSON to ./data:
  index.json          seasons, races, circuit pit-loss stats, model summary, fallback wear
  model.json          conditional-logit coefficients, scaler, held-out metrics, calibration
  s<year>.json        one file per season: lap times (ms), positions, pit stops, drivers,
                      neutralised laps, per-race tyre regression, per-race pit loss

Only 2011+ races that have both lap_times and pit_stops are used (329 races).

Tyre compounds and weather are not in the source data. Stints are inferred from pit stops.
Pit "milliseconds" is the time spent in the pit lane (Ergast calls it duration); this is the
lane time shown in the race feed.

Run:  python build_data.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
RAW_CIRCUITS = ROOT / "raw_circuits"
PORTRAIT_DIR = ROOT / "assets" / "drivers"

# Circuit geometry is fetched once during the build and cached under raw_circuits/.
# The browser only ever reads the compact paths written to data/index.json.
GEOJSON_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
SVG_INDEX_URL = "https://raw.githubusercontent.com/MisterSix2295/f1-circuits-svg/main/circuits.json"
SVG_BASE_URL = "https://raw.githubusercontent.com/MisterSix2295/f1-circuits-svg/main/circuits/black"
SVG_REF_ALIASES = {
    "albert_park": "melbourne", "villeneuve": "montreal", "interlagos": "interlagos",
    "nurburgring": "nurburgring", "red_bull_ring": "spielberg", "vegas": "las-vegas",
    "zandvoort": "zandvoort", "yeongam": "yeongam", "buddh": "buddh",
    "madring": None, "losail": "lusail", "portimao": "portimao",
    "ricard": "paul-ricard", "mugello": "mugello", "rodriguez": "mexico-city",
    "americas": "austin", "marina_bay": "marina-bay", "yas_marina": "yas-marina",
    "monaco": "monaco", "spa": "spa-francorchamps", "monza": "monza",
    "silverstone": "silverstone", "hockenheimring": "hockenheimring",
    "hungaroring": "hungaroring", "sepang": "sepang", "bahrain": "bahrain",
    "catalunya": "catalunya", "istanbul": "istanbul", "imola": "imola",
    "suzuka": "suzuka", "shanghai": "shanghai", "sochi": "sochi",
    "baku": "baku", "jeddah": "jeddah", "valencia": "valencia",
}
VIEW_W, VIEW_H, VIEW_PAD = 220.0, 130.0, 8.0
TARGET_POINTS = 110

# Official F1 constructor colours. The map covers every modern/legacy team ref in
# constructors.csv; engine-badged variants inherit their base constructor's colour
# via TEAM_TOKENS. Anything genuinely obscure gets a deterministic fallback below.
TEAM_COLORS = {
    # Current and modern era.
    "ferrari": "#E8002D", "red_bull": "#3671C6", "mercedes": "#27F4D2",
    "mclaren": "#FF8000", "aston_martin": "#229971", "alpine": "#00A1E8",
    "williams": "#1868DB", "rb": "#6692FF", "alphatauri": "#2B4562",
    "toro_rosso": "#469BFF", "sauber": "#52E252", "alfa": "#C92D4B",
    "alfa_romeo": "#C92D4B", "haas": "#B6BABD", "audi": "#F50537",
    "cadillac": "#909090", "renault": "#FFF500", "racing_point": "#F596C8",
    "force_india": "#FF5F0F", "bmw_sauber": "#0047AB", "toyota": "#D71920",
    "super_aguri": "#C8102E", "honda": "#E60012", "spyker": "#D4A017",
    "spyker_mf1": "#D4A017", "mf1": "#C8102E", "bar": "#002D62",
    "jordan": "#EED202", "benetton": "#00A86B", "brawn": "#B8FD6E",
    "brabham": "#002855", "tyrrell": "#003399", "matra": "#0055A4",
    "cooper": "#004225", "brm": "#004225", "vanwall": "#006633",
    "maserati": "#C8102E", "team_lotus": "#FFB800", "lotus_f1": "#FFB800",
    "lotus_racing": "#FFB800", "virgin": "#6F2DA8", "marussia": "#C8102E",
    "caterham": "#006B3F", "manor": "#6F2DA8", "hrt": "#D2232A",
    # Historic constructors with recognisable identities.
    "minardi": "#404040", "jaguar": "#006EAA", "prost": "#002F6C",
    "arrows": "#D2232A", "stewart": "#9E1B32", "lola": "#D7191C",
    "ligier": "#002395", "forti": "#D32F2F", "footwork": "#0033A0",
    "pacific": "#D2232A", "simtek": "#1D4ED8", "larrousse": "#F2C300",
    "dallara": "#003DA5", "fondmetal": "#B22222", "march": "#0057B8",
    "moda": "#C8102E", "ags": "#F2C300", "lambo": "#C8102E",
    "leyton": "#006B3F", "coloni": "#005A9C", "eurobrun": "#C8102E",
    "osella": "#C8102E", "onyx": "#003399", "life": "#C8102E",
    "rial": "#1D4ED8", "zakspeed": "#C8102E", "ram": "#1D4ED8",
    "spirit": "#1D4ED8", "toleman": "#D2232A", "ats": "#D2232A",
    "theodore": "#D2232A", "fittipaldi": "#006B3F", "ensign": "#C8102E",
    "shadow": "#D2232A", "wolf": "#C8102E", "merzario": "#C8102E",
    "surtees": "#C8102E", "hesketh": "#C8102E", "martini": "#D2232A",
    "penske": "#C8102E", "gordini": "#0055A4", "bugatti": "#C8102E",
    "lancia": "#C8102E", "bmw": "#0066B2", "porsche": "#C8102E",
    "eagle": "#C8102E", "de_tomaso": "#C8102E", "simca": "#0055A4",
    "lago": "#0055A4",
}
# Longest tokens first so de_tomaso-alfa_romeo resolves to De Tomaso, not Alfa.
TEAM_TOKENS = tuple(
    sorted(
        {
            "de_tomaso": "#C8102E", "alfa_romeo": "#C92D4B", "alfaromeo": "#C92D4B",
            "lotus": "#FFB800", "mclaren": "#FF8000", "brabham": "#002855",
            "cooper": "#004225", "matra": "#0055A4", "shadow": "#D2232A",
            "march": "#0057B8", "eagle": "#C8102E", "alfa": "#C92D4B",
            "porsche": "#C8102E", "ferrari": "#E8002D", "maserati": "#C8102E",
            "tomaso": "#C8102E", "brm": "#004225",
        }.items(),
        key=lambda kv: len(kv[0]), reverse=True,
    )
)
FALLBACK_TEAM_COLORS = (
    "#005A9C", "#B22222", "#006B3F", "#8A6D00",
    "#6A1B9A", "#00796B", "#A1006B", "#37474F",
)


def constructor_color(ref: str | None, name: str | None = None) -> str:
    """Official hex for a constructorRef, with a stable fallback for obscure teams."""
    key = str(ref or "")
    if key in TEAM_COLORS:
        return TEAM_COLORS[key]
    flat = re.sub(r"[^a-z0-9]+", "", key.lower())
    for token, color in TEAM_TOKENS:
        if re.sub(r"[^a-z0-9]+", "", token.lower()) in flat:
            return color
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return FALLBACK_TEAM_COLORS[digest[0] % len(FALLBACK_TEAM_COLORS)]


def all_constructor_colors(constructors: pd.DataFrame) -> dict[str, str]:
    """Every constructorRef in constructors.csv, including historical one-off entries."""
    return {
        str(row.constructorRef): constructor_color(str(row.constructorRef), str(row.name))
        for row in constructors.itertuples(index=False)
    }

# A stop with a larger lane time than this is not a normal green stop.
PIT_LOSS_MIN = 5.0
PIT_LOSS_MAX = 45.0

# Reliability gates for the per-race tyre regression. The paper spec requires the slope
# floor and neutralisation check. The extra gates catch obviously corrupted lap-time
# records (for example a lap that swallowed a red-flag stoppage) so the undercut model
# never uses a 40-second residual sd.
WEAR_MIN_SLOPE = 0.005
WEAR_MAX_SLOPE = 0.20
WEAR_MAX_SD = 2.0
WEAR_MIN_N = 100
WEAR_MAX_NEUTRAL_FRAC = 0.20

LAMBDA = 1e-3
RANDOM_SEED = 20261116


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name, na_values=["\\N"], low_memory=False)


def finite(x) -> bool:
    return x is not None and np.isfinite(x)


def rnd(x, nd=6):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


def median_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    return {
        "med": rnd(np.median(a), 3),
        "p25": rnd(np.percentile(a, 25), 3),
        "p75": rnd(np.percentile(a, 75), 3),
        "sd": rnd(np.std(a, ddof=1), 3) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def median_js(vals: list[float]) -> float:
    """Median that matches the JS implementation for the small pace windows."""
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return float("nan")
    if n % 2:
        return float(v[n // 2])
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def fmt_driver_code(row) -> str:
    code = row.get("code")
    if isinstance(code, str) and code.strip():
        return code.strip().upper()
    surname = str(row.get("surname", "")).strip().upper()
    clean = "".join(ch for ch in surname if ch.isalpha())
    return (clean[:3] or "???").ljust(3, "?")


def cached_download(url: str, path: Path, label: str) -> str | None:
    """Fetch once into raw_circuits/ so later builds (and the page) are offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path.read_text(encoding="utf-8")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PITWALL build/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            text = response.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"cached {label}: {path.name} ({len(text)} chars)")
        return text
    except Exception as exc:
        print(f"WARNING: could not fetch {label}: {exc}")
        return None


def norm_key(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def has_portrait(driver_ref) -> bool:
    if not driver_ref or not isinstance(driver_ref, str):
        return False
    return (PORTRAIT_DIR / f"{driver_ref}.png").is_file()


def ramer_douglas_peucker(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    ax, ay = points[0]
    bx, by = points[-1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    farthest, best = 0, 0.0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        dist = math.hypot(px - ax, py - ay) if den == 0 else abs(dy * px - dx * py + bx * ay - by * ax) / math.sqrt(den)
        if dist > best:
            farthest, best = i, dist
    if best > eps:
        return ramer_douglas_peucker(points[: farthest + 1], eps)[:-1] + ramer_douglas_peucker(points[farthest:], eps)
    return [points[0], points[-1]]


def simplify_track(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Binary-search the tolerance so each track stays near TARGET_POINTS vertices."""
    lo, hi = 0.02, 8.0
    best = ramer_douglas_peucker(points, lo)
    for _ in range(14):
        mid = (lo + hi) / 2.0
        trial = ramer_douglas_peucker(points, mid)
        if abs(len(trial) - TARGET_POINTS) < abs(len(best) - TARGET_POINTS):
            best = trial
        if len(trial) > TARGET_POINTS:
            lo = mid
        else:
            hi = mid
    if len(best) > 150:
        step = math.ceil(len(best) / 150)
        best = best[::step]
    if best[0] != best[-1]:
        best.append(best[0])
    return best


def fit_to_viewbox(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Uniform scaling, centred in the 220x130 viewBox with a small margin."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    sx = (VIEW_W - 2 * VIEW_PAD) / w if w > 0 else 0.0
    sy = (VIEW_H - 2 * VIEW_PAD) / h if h > 0 else 0.0
    scale = min(sx, sy) if sx > 0 and sy > 0 else 1.0
    ox = (VIEW_W - w * scale) / 2.0 - min(xs) * scale
    oy = (VIEW_H - h * scale) / 2.0 - min(ys) * scale
    return [(round(x * scale + ox, 2), round(y * scale + oy, 2)) for x, y in points]


def project_lat_lng(coords: list[list[float]]) -> list[tuple[float, float]]:
    lat0 = math.radians(float(np.median([c[1] for c in coords])))
    factor = math.cos(lat0)
    return [(float(c[0]) * factor, float(c[1])) for c in coords]


def sf_tick(points: list[tuple[float, float]]) -> list[float]:
    x0, y0 = points[0]
    x1, y1 = next((p for p in points[1:] if p != (x0, y0)), (x0 + 1.0, y0))
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length * 5.0, dx / length * 5.0
    return [round(x0 - nx, 2), round(y0 - ny, 2), round(x0 + nx, 2), round(y0 + ny, 2)]


def layout_from_points(points: list[tuple[float, float]], source: str) -> dict:
    fitted = fit_to_viewbox(points)
    simple = simplify_track(fitted)
    path = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in simple) + " Z"
    return {"svg_path": path, "sf_line": sf_tick(simple), "points": len(simple), "source": source}


def parse_svg_path(data: str) -> list[tuple[float, float]]:
    """Flatten the M/h/l/s/c/q SVG commands used by the cached layout files."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", data)
    pts: list[tuple[float, float]] = []
    i = 0
    cur = start = (0.0, 0.0)
    prev_c2 = None
    cmd = None

    def take(n: int) -> list[float] | None:
        nonlocal i
        if i + n > len(tokens):
            return None
        out = []
        for _ in range(n):
            if re.match(r"^[A-Za-z]$", tokens[i]):
                return None
            out.append(float(tokens[i]))
            i += 1
        return out

    def relpair(vals: list[float]) -> tuple[float, float]:
        return (vals[0] + cur[0], vals[1] + cur[1]) if cmd and cmd.islower() else (vals[0], vals[1])

    def cubic(c1, c2, end, samples=5):
        nonlocal cur
        for k in range(1, samples + 1):
            t = k / samples
            u = 1.0 - t
            pts.append((
                u ** 3 * cur[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * end[0],
                u ** 3 * cur[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * end[1],
            ))

    while i < len(tokens):
        if re.match(r"^[A-Za-z]$", tokens[i]):
            cmd = tokens[i]
            i += 1
        if not cmd:
            break
        upper = cmd.upper()
        if upper == "Z":
            pts.append(start)
            cur = start
            cmd = None
            continue
        if upper == "M":
            vals = take(2)
            if vals is None:
                break
            cur = relpair(vals)
            start = cur
            pts.append(cur)
            cmd = "l" if cmd.islower() else "L"
            continue
        if upper == "L":
            vals = take(2)
            if vals is None:
                break
            cur = relpair(vals)
            pts.append(cur)
            prev_c2 = None
        elif upper in ("H", "V"):
            vals = take(1)
            if vals is None:
                break
            v = vals[0] + (cur[0] if upper == "H" else cur[1]) if cmd.islower() else vals[0]
            cur = (v, cur[1]) if upper == "H" else (cur[0], v)
            pts.append(cur)
            prev_c2 = None
        elif upper == "C":
            vals = take(6)
            if vals is None:
                break
            pairs = [relpair(vals[j:j + 2]) for j in (0, 2, 4)]
            cubic(pairs[0], pairs[1], pairs[2])
            prev_c2, cur = pairs[1], pairs[2]
        elif upper == "S":
            vals = take(4)
            if vals is None:
                break
            c2, end = relpair(vals[0:2]), relpair(vals[2:4])
            c1 = (2 * cur[0] - prev_c2[0], 2 * cur[1] - prev_c2[1]) if prev_c2 else cur
            cubic(c1, c2, end)
            prev_c2, cur = c2, end
        elif upper == "Q":
            vals = take(4)
            if vals is None:
                break
            q, end = relpair(vals[0:2]), relpair(vals[2:4])
            c1 = (cur[0] + 2 * (q[0] - cur[0]) / 3, cur[1] + 2 * (q[1] - cur[1]) / 3)
            c2 = (end[0] + 2 * (q[0] - end[0]) / 3, end[1] + 2 * (q[1] - end[1]) / 3)
            cubic(c1, c2, end)
            prev_c2, cur = q, end
        elif upper == "T":
            vals = take(2)
            if vals is None:
                break
            end = relpair(vals)
            q = (2 * cur[0] - prev_c2[0], 2 * cur[1] - prev_c2[1]) if prev_c2 else cur
            c1 = (cur[0] + 2 * (q[0] - cur[0]) / 3, cur[1] + 2 * (q[1] - cur[1]) / 3)
            c2 = (end[0] + 2 * (q[0] - end[0]) / 3, end[1] + 2 * (q[1] - end[1]) / 3)
            cubic(c1, c2, end)
            prev_c2, cur = q, end
        elif upper == "A":
            vals = take(7)
            if vals is None:
                break
            rx, ry, angle, large, sweep = vals[:5]
            end = relpair(vals[5:7])
            x0, y0 = cur
            if rx == 0 or ry == 0 or end == cur:
                cur = end
                pts.append(cur)
                prev_c2 = None
                continue
            phi = math.radians(angle % 180.0)
            cosp, sinp = math.cos(phi), math.sin(phi)
            dx2, dy2 = (x0 - end[0]) / 2.0, (y0 - end[1]) / 2.0
            x1p, y1p = cosp * dx2 + sinp * dy2, -sinp * dx2 + cosp * dy2
            scale = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
            if scale > 1.0:
                rx *= math.sqrt(scale)
                ry *= math.sqrt(scale)
            num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
            den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
            coef = math.sqrt(max(0.0, num / den)) if den else 0.0
            sign = -1.0 if bool(large) == bool(sweep) else 1.0
            cxp, cyp = sign * coef * rx * y1p / ry, -sign * coef * ry * x1p / rx
            cx = cosp * cxp - sinp * cyp + (x0 + end[0]) / 2.0
            cy = sinp * cxp + cosp * cyp + (y0 + end[1]) / 2.0
            theta1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
            theta2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
            delta = theta2 - theta1
            if sweep == 0 and delta > 0:
                delta -= 2 * math.pi
            elif sweep != 0 and delta < 0:
                delta += 2 * math.pi
            for k in range(1, 6):
                theta = theta1 + delta * k / 5.0
                pts.append((
                    cosp * rx * math.cos(theta) - sinp * ry * math.sin(theta) + cx,
                    sinp * rx * math.cos(theta) + cosp * ry * math.sin(theta) + cy,
                ))
            cur = end
            prev_c2 = None
        else:
            break
    return pts


def build_circuit_layouts(circuits: pd.DataFrame, replay_ids: set[int]) -> dict:
    """GeoJSON layouts plus cached era-specific SVG layouts, keyed by circuitId."""
    geo_text = cached_download(GEOJSON_URL, RAW_CIRCUITS / "f1-circuits.geojson", "circuit GeoJSON")
    svg_text = cached_download(SVG_INDEX_URL, RAW_CIRCUITS / "svg-circuits.json", "SVG circuit index")
    features = json.loads(geo_text).get("features", []) if geo_text else []
    svg_index = {c["id"]: c for c in json.loads(svg_text)} if svg_text else {}
    geo_by_id = {str(f["properties"].get("id", "")): f for f in features}
    geo_by_name = {norm_key(f["properties"].get("Name")): f for f in features}
    geo_by_location = {norm_key(f["properties"].get("Location")): f for f in features}
    geo_aliases = {
        "albert_park": "au-1953", "villeneuve": "ca-1978", "interlagos": "br-1940",
        "nurburgring": "de-1927", "red_bull_ring": "at-1969", "vegas": "us-2023",
        "madring": "es-2026", "losail": "qa-2004", "portimao": "pt-2008",
        "ricard": "fr-1969", "mugello": "it-1914", "rodriguez": "mx-1962",
        "americas": "us-2012", "marina_bay": "sg-2008", "yas_marina": "ae-2009",
        "monaco": "mc-1929", "spa": "be-1925", "monza": "it-1922",
        "silverstone": "gb-1948", "hockenheimring": "de-1932", "hungaroring": "hu-1986",
        "sepang": "my-1999", "bahrain": "bh-2002", "catalunya": "es-1991",
        "istanbul": "tr-2005", "imola": "it-1953", "suzuka": "jp-1962",
        "shanghai": "cn-2004", "sochi": "ru-2014", "baku": "az-2016",
        "jeddah": "sa-2021", "buddh": None, "yeongam": None, "valencia": None,
    }
    out: dict[str, dict] = {}
    total_eras = 0
    for row in circuits[circuits.circuitId.isin(replay_ids)].itertuples(index=False):
        cid = int(row.circuitId)
        ref = str(row.circuitRef)
        feature = geo_by_id.get(geo_aliases.get(ref, "")) or geo_by_name.get(norm_key(row.name))
        if feature is None:
            feature = geo_by_location.get(norm_key(row.location))
        primary = None
        if feature and feature.get("geometry", {}).get("coordinates"):
            coords = feature["geometry"]["coordinates"]
            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                coords = coords[0]
            points = project_lat_lng(coords)
            if len(points) >= 4:
                primary = layout_from_points(points, "bacinger/f1-circuits GeoJSON")
        eras = []
        svg_id = SVG_REF_ALIASES.get(ref, norm_key(ref))
        svg_circuit = svg_index.get(svg_id)
        if svg_circuit:
            for layout in svg_circuit.get("layouts", []):
                layout_id = str(layout["layoutId"])
                cached = cached_download(
                    f"{SVG_BASE_URL}/{layout_id}.svg",
                    RAW_CIRCUITS / "svg" / f"{layout_id}.svg",
                    f"layout {layout_id}",
                )
                match = re.search(r'<path[^>]*\bd="([^"]+)"', cached or "")
                if not match:
                    continue
                points = parse_svg_path(match.group(1))
                if len(points) < 4:
                    continue
                season_years = [int(v) for v in re.findall(r"\d{4}", str(layout["seasons"]))]
                if not season_years:
                    continue
                entry = {"from": min(season_years), "to": max(season_years)}
                entry.update(layout_from_points(points, "f1-circuits-svg"))
                entry["layoutId"] = layout_id
                eras.append(entry)
        if not primary and eras:
            latest = eras[-1]
            primary = {k: latest[k] for k in ("svg_path", "sf_line", "points", "source")}
        out[str(cid)] = {
            "id": cid,
            "ref": ref,
            "name": str(row.name),
            "location": str(row.location),
            "country": str(row.country),
            "fallback": primary is None,
            **(primary or {"svg_path": "", "sf_line": [], "points": 0, "source": "unavailable"}),
            "eras": eras,
        }
        total_eras += len(eras)
    print(
        f"circuit layouts: {sum(1 for v in out.values() if not v['fallback'])}/{len(out)} tracks, "
        f"{total_eras} era variants"
    )
    return out


def build_champions(races, standings_d, standings_c, results, drivers, constructors, seasons_out):
    """Final-round champions for every season in the archive, 1950-2026."""
    drv = drivers.set_index("driverId")
    con = constructors.set_index("constructorId")
    race_year = races[["raceId", "year", "round"]].copy()
    finals = race_year.sort_values(["year", "round"]).groupby("year").tail(1)
    joined = results.merge(race_year[["raceId", "year"]], on="raceId", how="left")
    pos_order = pd.to_numeric(joined.positionOrder, errors="coerce")
    joined = joined.assign(pos=pos_order)
    team_by_driver = (
        joined[joined.pos.notna()]
        .groupby(["year", "driverId"])["constructorId"]
        .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0]))
        .to_dict()
    )
    driver_wins = joined[joined.pos == 1].groupby(["year", "driverId"]).size().to_dict()
    driver_podiums = joined[joined.pos <= 3].groupby(["year", "driverId"]).size().to_dict()
    constructor_wins = joined[joined.pos == 1].groupby(["year", "constructorId"]).size().to_dict()

    sd = standings_d.merge(race_year, on="raceId", how="left")
    sc = standings_c.merge(race_year, on="raceId", how="left")
    sd["pos_num"] = pd.to_numeric(sd.position, errors="coerce")
    sc["pos_num"] = pd.to_numeric(sc.position, errors="coerce")

    replay_years = {int(s["year"]): s for s in seasons_out}
    out = []
    for year in sorted(race_year.year.unique()):
        year = int(year)
        final_race = finals[finals.year == year].iloc[0]
        dyear = sd[sd.year == year]
        cyear = sc[sc.year == year]
        # Some seasons (notably the incomplete 2026 archive) stop before the last round.
        dround = int(dyear["round"].max()) if len(dyear) else None
        cround = int(cyear["round"].max()) if len(cyear) else None
        dtop = dyear[dyear["round"] == dround].sort_values(["pos_num", "points"], ascending=[True, False]) if dround else dyear.iloc[0:0]
        dnext = dtop[dtop.pos_num == 2]
        dchamp = dtop[dtop.pos_num == 1]
        ctop = cyear[cyear["round"] == cround].sort_values(["pos_num", "points"], ascending=[True, False]) if cround else cyear.iloc[0:0]
        cchamp = ctop[ctop.pos_num == 1]
        driver = None
        if len(dchamp):
            row = dchamp.iloc[0]
            did = int(row.driverId)
            drow = drv.loc[did] if did in drv.index else None
            ref = str(drow.driverRef) if drow is not None else None
            team_id = team_by_driver.get((year, did))
            crow = con.loc[team_id] if team_id in con.index else None
            team = str(crow["name"]) if crow is not None else "—"
            team_ref = str(crow["constructorRef"]) if crow is not None else None
            team_color = constructor_color(team_ref, team)
            points = rnd(float(row.points), 1) if pd.notna(row.points) else None
            margin = None
            if points is not None and len(dnext):
                runner_points = float(dnext.iloc[0].points) if pd.notna(dnext.iloc[0].points) else None
                if runner_points is not None:
                    margin = rnd(points - runner_points, 1)
            driver = {
                "driverId": did,
                "driverRef": ref,
                "code": fmt_driver_code(drow) if drow is not None else "—",
                "name": f"{drow.forename} {drow.surname}".strip() if drow is not None else f"Driver {did}",
                "nationality": str(drow.nationality) if drow is not None else "",
                "constructor": team,
                "constructorRef": team_ref,
                "color": team_color,
                "points": points,
                "wins": int(driver_wins.get((year, did), 0)),
                "podiums": int(driver_podiums.get((year, did), 0)),
                "margin": margin,
                "has_img": has_portrait(ref),
            }
        constructor = None
        if len(cchamp):
            row = cchamp.iloc[0]
            cid = int(row.constructorId)
            crow = con.loc[cid] if cid in con.index else None
            ref = str(crow["constructorRef"]) if crow is not None else None
            cname = str(crow["name"]) if crow is not None else f"Constructor {cid}"
            constructor = {
                "constructorId": cid,
                "constructorRef": ref,
                "name": cname,
                "nationality": str(crow["nationality"]) if crow is not None else "",
                "color": constructor_color(ref, cname),
                "points": rnd(float(row.points), 1) if pd.notna(row.points) else None,
                "wins": int(constructor_wins.get((year, cid), 0)),
            }
        replay = replay_years.get(year)
        default_race_id = int(replay["races"][-1]["id"]) if replay and replay["races"] else None
        out.append({
            "year": year,
            "driver": driver,
            "constructor": constructor,
            "has_replay": bool(replay),
            "default_race_id": default_race_id,
            "standings_race_id": int(final_race.raceId) if (dround == int(final_race["round"]) or cround == int(final_race["round"])) else None,
            "standings_round": dround,
        })
    return out


def process_race(
    rid: int,
    race_row,
    x: pd.DataFrame,
    pits_df: pd.DataFrame,
    res_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    constructors_df: pd.DataFrame,
    qual_df: pd.DataFrame,
    last_team: dict[int, int],
):
    """Return the race JSON object and model rows for one race."""
    year = int(race_row.year)
    circuit_id = int(race_row.circuitId)

    driver_ids = sorted(int(d) for d in x.driverId.unique())
    idx = {d: i for i, d in enumerate(driver_ids)}
    n_drivers = len(driver_ids)
    max_lap = int(x.lap.max())

    lap_ms = np.full((n_drivers, max_lap), np.nan, dtype=float)
    pos = np.full((n_drivers, max_lap), np.nan, dtype=float)
    for row in x.itertuples(index=False):
        i = idx[int(row.driverId)]
        lap = int(row.lap)
        if 1 <= lap <= max_lap:
            lap_ms[i, lap - 1] = float(row.milliseconds)
            pos[i, lap - 1] = float(row.position)

    pits: dict[int, list[tuple[int, float | None]]] = {d: [] for d in driver_ids}
    inout = np.zeros((n_drivers, max_lap), dtype=bool)
    for row in pits_df.itertuples(index=False):
        d = int(row.driverId)
        if d not in idx:
            continue
        i = idx[d]
        lap = int(row.lap)
        lane = None if pd.isna(row.milliseconds) else float(row.milliseconds)
        pits[d].append((lap, lane))
        if 1 <= lap <= max_lap:
            inout[i, lap - 1] = True
        if 2 <= lap + 1 <= max_lap:
            inout[i, lap] = True
    for d in pits:
        pits[d].sort(key=lambda z: z[0])

    # ---- 1. neutralised laps -------------------------------------------------
    # Field median lap time per lap, excluding in/out laps. Lap 1 is never used.
    field_med: dict[int, float] = {}
    for lap in range(2, max_lap + 1):
        col = lap_ms[:, lap - 1]
        mask = np.isfinite(col) & ~inout[:, lap - 1]
        if int(mask.sum()) >= 5:
            field_med[lap] = float(np.median(col[mask]))
    if field_med:
        p15 = float(np.percentile(np.asarray(list(field_med.values())), 15))
        threshold = 1.08 * p15
        neutral = sorted(lap for lap, med in field_med.items() if med > threshold)
    else:
        neutral = []
    neutral_set = set(neutral)
    neutral_frac = len(neutral) / max(1, max_lap - 1)

    # ---- 2. tyre wear --------------------------------------------------------
    # OLS of lap time on [stint age, race lap] with driver fixed effects.
    # Demeaning per driver absorbs the intercept.
    clean_rows: list[tuple[int, int, int, float]] = []
    for i, d in enumerate(driver_ids):
        pit_laps = [lap for lap, _ in pits[d] if 1 <= lap < max_lap]
        starts = [1] + [lap + 1 for lap in pit_laps]
        ends = pit_laps + [max_lap]
        age = np.zeros(max_lap, dtype=float)
        for s, e in zip(starts, ends):
            if 1 <= s <= e <= max_lap:
                age[s - 1 : e] = np.arange(1, e - s + 2)
        for lap in range(2, max_lap + 1):
            if not np.isfinite(lap_ms[i, lap - 1]):
                continue
            if lap in neutral_set or inout[i, lap - 1]:
                continue
            clean_rows.append((i, int(age[lap - 1]), lap, float(lap_ms[i, lap - 1])))

    slope = fuel = r2 = sd = 0.0
    wear_n = len(clean_rows)
    wear_drivers = 0
    if wear_n >= 10:
        cdf = pd.DataFrame(clean_rows, columns=["i", "age", "lap", "ms"])
        wear_drivers = int(cdf["i"].nunique())
        dm = cdf.copy()
        for col in ("age", "lap", "ms"):
            dm[col] = cdf[col] - cdf.groupby("i")[col].transform("mean")
        X = np.column_stack([dm["age"].to_numpy(float), dm["lap"].to_numpy(float)])
        y = dm["ms"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sst = float((y ** 2).sum())
        sse = float((resid ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        dof = max(1, wear_n - wear_drivers - 2)
        sd = float(np.sqrt(sse / dof)) / 1000.0
        slope = float(beta[0] / 1000.0)
        fuel = float(beta[1] / 1000.0)

    wear_reasons = []
    if wear_n < WEAR_MIN_N:
        wear_reasons.append(f"n={wear_n} below {WEAR_MIN_N}")
    if neutral_frac > WEAR_MAX_NEUTRAL_FRAC:
        wear_reasons.append(f"neutralised {neutral_frac * 100:.0f}% of laps")
    if slope < WEAR_MIN_SLOPE:
        wear_reasons.append(f"slope {slope:.3f} s/lap below {WEAR_MIN_SLOPE:.3f}")
    if slope > WEAR_MAX_SLOPE:
        wear_reasons.append(f"slope {slope:.3f} s/lap above {WEAR_MAX_SLOPE:.3f}")
    if sd > WEAR_MAX_SD:
        wear_reasons.append(f"residual sd {sd:.2f} s above {WEAR_MAX_SD:.2f}")
    wear_reliable = len(wear_reasons) == 0

    # ---- 3. pit loss ---------------------------------------------------------
    # (in-lap + out-lap) - 2 * median of the 3 clean laps before. Green stops only.
    loss_values: list[float] = []
    for d in driver_ids:
        i = idx[d]
        for lap, _lane in pits[d]:
            if lap < 2 or lap >= max_lap:
                continue
            if lap in neutral_set or (lap + 1) in neutral_set:
                continue
            in_ms = lap_ms[i, lap - 1]
            out_ms = lap_ms[i, lap]
            if not (np.isfinite(in_ms) and np.isfinite(out_ms)):
                continue
            before: list[float] = []
            for ll in range(lap - 1, 1, -1):
                if ll in neutral_set:
                    continue
                if inout[i, ll - 1]:
                    continue
                if np.isfinite(lap_ms[i, ll - 1]):
                    before.append(float(lap_ms[i, ll - 1]))
                if len(before) == 3:
                    break
            if len(before) < 3:
                continue
            loss = (float(in_ms) + float(out_ms) - 2 * float(np.median(before))) / 1000.0
            if PIT_LOSS_MIN < loss < PIT_LOSS_MAX:
                loss_values.append(loss)
    race_pit = median_stats(loss_values)

    # ---- driver metadata / teams -------------------------------------------
    res_by_driver = {}
    if len(res_df):
        for row in res_df.itertuples(index=False):
            res_by_driver[int(row.driverId)] = row
    qual_by_driver = {}
    qual_by_number = {}
    if len(qual_df):
        # Join by the driver's season car number from drivers.csv, not the
        # qualifying row's number: the latter is missing for some late-season
        # substitute entries (e.g. Camara, 2026 Brazil).
        driver_numbers = {}
        for drow in drivers_df.itertuples(index=False):
            try:
                driver_numbers[int(drow.driverId)] = int(drow.number)
            except (TypeError, ValueError):
                pass
        for row in qual_df.itertuples(index=False):
            qual_by_driver[int(row.driverId)] = row
            num = driver_numbers.get(int(row.driverId))
            if num is not None and num not in qual_by_number:
                qual_by_number[num] = row

    season_driver_meta = {}
    race_drivers = []
    pit_loss_circuit_values = []

    for i, d in enumerate(driver_ids):
        drow = drivers_df.loc[drivers_df.driverId == d]
        ref = None
        if len(drow):
            dr = drow.iloc[0]
            code = fmt_driver_code(dr)
            name = f"{dr.forename} {dr.surname}".strip()
            ref = str(dr.driverRef)
        else:
            code = "???"
            name = f"Driver {d}"
        season_driver_meta[str(d)] = {"code": code, "name": name, "driverRef": ref}

        res_row = res_by_driver.get(d)
        no_result = res_row is None
        team_id = None
        team_source = "results"
        if res_row is not None:
            team_id = int(res_row.constructorId)
        else:
            qrow = qual_by_driver.get(d)
            if qrow is not None:
                team_id = int(qrow.constructorId)
                team_source = "qualifying"
            else:
                # 2026 Brazilian GP: Fittipaldi has laps but neither a result nor a
                # qualifying row. His car number matches the qualifying entry of the
                # driver he replaced, so the team comes from qualifying by number.
                num = None
                if len(drow):
                    try:
                        num = int(drow.iloc[0].number)
                    except (TypeError, ValueError):
                        num = None
                if num is not None and num in qual_by_number:
                    team_id = int(qual_by_number[num].constructorId)
                    team_source = f"qualifying (car No. {num})"
                elif d in last_team:
                    team_id = int(last_team[d])
                    team_source = "most recent result"
                else:
                    team_source = "unknown"
        team = "Unknown"
        team_ref = None
        team_color = constructor_color(None, team)
        if team_id is not None:
            crow = constructors_df.loc[constructors_df.constructorId == team_id]
            if len(crow):
                team = str(crow.iloc[0]["name"])
                team_ref = str(crow.iloc[0]["constructorRef"])
                team_color = constructor_color(team_ref, team)
        season_driver_meta[str(d)]["constructorRef"] = team_ref
        season_driver_meta[str(d)]["color"] = team_color

        result = None
        if res_row is not None:
            pos_text = str(res_row.positionText) if not pd.isna(res_row.positionText) else ""
            ret = pos_text.upper() == "R"
            status = str(res_row.status) if not pd.isna(res_row.status) else "Unknown"
            grid = None if pd.isna(res_row.grid) else int(res_row.grid)
            final_pos = None if pd.isna(res_row.positionOrder) else int(res_row.positionOrder)
            result = {"pos": final_pos, "grid": grid, "ret": ret}
            if ret:
                result["st"] = status

        laps = [int(v) for v in lap_ms[i, :] if np.isfinite(v)]
        positions = [int(v) for v in pos[i, :] if np.isfinite(v)]
        # Arrays are contiguous from lap 1. Guard anyway.
        first_lap = 1
        if len(laps) != len(positions):
            cut = min(len(laps), len(positions))
            laps, positions = laps[:cut], positions[:cut]
        race_drivers.append(
            {
                "id": d,
                "team": team,
                "noResult": bool(no_result),
                "teamSource": team_source,
                "result": result,
                "laps": laps,
                "pos": positions,
                "pits": [[int(l), rnd(lane, 0) if lane is not None else None] for l, lane in pits[d]],
            }
        )

        if d in last_team:
            pass
        if res_row is not None:
            last_team[d] = int(res_row.constructorId)
        pit_loss_circuit_values.extend([])

    # winner
    winner_id = None
    if len(res_df):
        w = res_df[res_df.positionOrder == 1]
        if len(w):
            winner_id = int(w.iloc[0].driverId)

    # ---- model feature rows -------------------------------------------------
    # Recent pace: last up to five clean laps, looking back at most ten laps.
    pace_ms = np.full((n_drivers, max_lap), np.nan, dtype=float)
    for i in range(n_drivers):
        for lap in range(1, max_lap + 1):
            vals: list[float] = []
            for ll in range(lap, max(0, lap - 10), -1):
                if ll < 1:
                    break
                if (
                    ll >= 2
                    and ll not in neutral_set
                    and not inout[i, ll - 1]
                    and np.isfinite(lap_ms[i, ll - 1])
                ):
                    vals.append(float(lap_ms[i, ll - 1]))
                if len(vals) == 5:
                    break
            if len(vals) < 2:
                vals = []
                for ll in range(lap, 0, -1):
                    if np.isfinite(lap_ms[i, ll - 1]):
                        vals.append(float(lap_ms[i, ll - 1]))
                    if len(vals) == 5:
                        break
            if vals:
                pace_ms[i, lap - 1] = median_js(vals)

    cum_ms = np.full((n_drivers, max_lap), np.nan, dtype=float)
    for i in range(n_drivers):
        acc = 0.0
        for lap in range(1, max_lap + 1):
            if not np.isfinite(lap_ms[i, lap - 1]):
                break
            acc += float(lap_ms[i, lap - 1])
            cum_ms[i, lap - 1] = acc

    stops_before = np.zeros((n_drivers, max_lap + 1), dtype=int)
    for i, d in enumerate(driver_ids):
        plist = [lap for lap, _ in pits[d]]
        for lap in range(1, max_lap + 1):
            stops_before[i, lap] = int(np.searchsorted(plist, lap, side="left"))

    model_rows = []
    lead_changes = 0
    prev_leader = None
    leaders_by_lap = {}
    probs_later = {}
    for lap in range(1, max_lap + 1):
        running = [i for i in range(n_drivers) if np.isfinite(lap_ms[i, lap - 1])]
        if len(running) < 2:
            continue
        leader = min(running, key=lambda i: pos[i, lap - 1])
        leaders_by_lap[lap] = driver_ids[leader]
        if prev_leader is not None and driver_ids[leader] != prev_leader:
            lead_changes += 1
        prev_leader = driver_ids[leader]
        if winner_id is None or winner_id not in [driver_ids[i] for i in running]:
            continue
        winner_i = idx[winner_id]
        pace_lead = pace_ms[leader, lap - 1]
        stop_counts = [int(stops_before[i, lap]) for i in running]
        med_stops = float(np.median(stop_counts))
        laps_left = max_lap - lap
        for i in running:
            gap_s = max(0.0, (cum_ms[i, lap - 1] - cum_ms[leader, lap - 1]) / 1000.0)
            pace = pace_ms[i, lap - 1]
            if not np.isfinite(pace) or not np.isfinite(pace_lead):
                pace_diff = 0.0
            else:
                pace_diff = float(np.clip((pace - pace_lead) / 1000.0, -3.0, 5.0))
            f1 = math.log(float(pos[i, lap - 1]))
            f2 = f1 * laps_left / max_lap
            f3 = min(gap_s, 300.0) / math.sqrt(laps_left + 1)
            f4 = pace_diff * math.sqrt(laps_left + 1) / 10.0
            f5 = float(np.clip(stops_before[i, lap] - med_stops, -2.0, 2.0))
            model_rows.append(
                {
                    "rid": rid,
                    "lap": lap,
                    "d": driver_ids[i],
                    "pos": int(pos[i, lap - 1]),
                    "y": 1 if i == winner_i else 0,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "f4": f4,
                    "f5": f5,
                }
            )

    race_json = {
        "id": rid,
        "round": int(race_row["round"]),
        "name": str(race_row["name"]),
        "date": str(race_row.date),
        "circuitId": circuit_id,
        "circuit": str(race_row.circuit_name),
        "country": str(race_row.country),
        "laps": max_lap,
        "drivers": race_drivers,
        "neutral": neutral,
        "wear": {
            "slope": rnd(slope, 6),
            "fuel": rnd(fuel, 6),
            "r2": rnd(r2, 5),
            "sd": rnd(sd, 4),
            "n": int(wear_n),
            "reliable": bool(wear_reliable),
            "reason": "; ".join(wear_reasons),
        },
        "pitLoss": race_pit,
    }
    meta = {
        "year": year,
        "total": max_lap,
        "winner": winner_id,
        "leadChanges": lead_changes,
        "neutralFrac": neutral_frac,
        "leadersByLap": leaders_by_lap,
        "losses": loss_values,
        "wearSlope": slope,
        "wearSd": sd,
        "wearReliable": wear_reliable,
        "wearN": wear_n,
        "wearR2": r2,
        "neutral": neutral,
    }
    return race_json, season_driver_meta, model_rows, meta


def sort_groups(groups: np.ndarray, *arrays: np.ndarray):
    order = np.argsort(groups, kind="stable")
    return (groups[order],) + tuple(a[order] for a in arrays)


def group_starts(groups: np.ndarray) -> np.ndarray:
    if len(groups) == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])


def fit_conditional_logit(X: np.ndarray, groups: np.ndarray, y: np.ndarray, lam: float = LAMBDA):
    groups, X, y = sort_groups(groups, X, y)
    starts = group_starts(groups)
    G = len(starts)
    sizes = np.diff(np.r_[starts, len(groups)])
    win_idx = np.flatnonzero(y == 1)
    if len(win_idx) != G:
        raise ValueError(f"expected one winner per group: {len(win_idx)} winners, {G} groups")
    win_group = groups[win_idx]

    def fg(beta):
        s = X @ beta
        gmax = np.maximum.reduceat(s, starts)
        ex = np.exp(s - gmax[groups])
        gsum = np.add.reduceat(ex, starts)
        logz = gmax + np.log(gsum)
        loss = -float(np.sum(s[win_idx] - logz[win_group])) / G + 0.5 * lam * float(beta @ beta)
        p = ex / gsum[groups]
        grad = (X * p[:, None]).sum(axis=0) - X[win_idx].sum(axis=0)
        grad = grad / G + lam * beta
        return loss, grad

    res = minimize(
        lambda b: fg(b),
        np.zeros(X.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50},
    )
    return res


def evaluate(X: np.ndarray, groups: np.ndarray, y: np.ndarray, pos: np.ndarray, beta: np.ndarray):
    groups, X, y, pos = sort_groups(groups, X, y, pos)
    starts = group_starts(groups)
    G = len(starts)
    s = X @ beta
    gmax = np.maximum.reduceat(s, starts)
    ex = np.exp(s - gmax[groups])
    gsum = np.add.reduceat(ex, starts)
    p = ex / gsum[groups]
    win_idx = np.flatnonzero(y == 1)
    logloss = -float(np.mean(np.log(np.clip(p[win_idx], 1e-300, None))))
    brier = float(np.mean(np.add.reduceat((p - y) ** 2, starts)))
    win_local = np.empty(G, dtype=int)
    win_local[groups[win_idx]] = win_idx - starts[groups[win_idx]]
    correct = 0
    for g in range(G):
        a = starts[g]
        b = starts[g + 1] if g + 1 < G else len(p)
        if int(np.argmax(p[a:b])) == int(win_local[g]):
            correct += 1
    accuracy = correct / G
    return {
        "logloss": logloss,
        "brier": brier,
        "accuracy": accuracy,
        "p": p,
        "starts": starts,
        "sizes": np.diff(np.r_[starts, len(groups)]),
        "win_idx": win_idx,
        "groups_sorted": groups,
        "y_sorted": y,
        "pos_sorted": pos,
    }


def calibration_bins(p: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            out.append(
                {
                    "p": rnd(float(p[mask].mean()), 6),
                    "obs": rnd(float(y[mask].mean()), 6),
                    "n": n,
                }
            )
        else:
            out.append({"p": None, "obs": None, "n": 0})
    return out


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    races = read_csv("races.csv")
    lap_times = read_csv("lap_times.csv")
    pit_stops = read_csv("pit_stops.csv")
    results = read_csv("results.csv")
    drivers = read_csv("drivers.csv")
    constructors = read_csv("constructors.csv")
    circuits = read_csv("circuits.csv")
    status = read_csv("status.csv")
    qualifying = read_csv("qualifying.csv")
    driver_standings = read_csv("driver_standings.csv")
    constructor_standings = read_csv("constructor_standings.csv")

    lap_times["racerId"] = lap_times.raceId
    lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")
    lap_times["position"] = pd.to_numeric(lap_times["position"], errors="coerce")
    pit_stops["milliseconds"] = pd.to_numeric(pit_stops["milliseconds"], errors="coerce")
    pit_stops["lap"] = pd.to_numeric(pit_stops["lap"], errors="coerce")

    results = results.merge(status, on="statusId", how="left")
    circuits_small = circuits.rename(
        columns={"name": "circuit_name", "url": "circuit_url", "alt": "circuit_alt"}
    )[["circuitId", "circuit_name", "location", "country"]]
    race_circuits = races.merge(circuits_small, on="circuitId", how="left")

    lap_ids = set(int(v) for v in lap_times.raceId.unique())
    pit_ids = set(int(v) for v in pit_stops.raceId.unique())
    valid = race_circuits[
        (race_circuits.year >= 2011)
        & (race_circuits.raceId.isin(lap_ids))
        & (race_circuits.raceId.isin(pit_ids))
    ].sort_values(["year", "round"])
    valid = valid.copy()
    race_ids = [int(v) for v in valid.raceId]
    ref_by_id = {int(r.driverId): str(r.driverRef) for r in drivers.itertuples(index=False)}
    print(f"races: {len(race_ids)}  years: {valid.year.min()}-{valid.year.max()}")

    lap_by_race = {int(k): v for k, v in lap_times.groupby("raceId")}
    pit_by_race = {int(k): v for k, v in pit_stops.groupby("raceId")}
    res_by_race = {int(k): v for k, v in results.groupby("raceId")}
    qual_by_race = {int(k): v for k, v in qualifying.groupby("raceId")}

    # Last known team for a driver, used only if both result and qualifying are absent.
    # Updated at the end of each race so the fallback only looks backwards.
    last_team: dict[int, int] = {}

    seasons: dict[int, dict] = defaultdict(lambda: {"year": 0, "drivers": {}, "races": []})
    circuit_losses: dict[int, list[float]] = defaultdict(list)
    wear_all: list[dict] = []
    model_records: list[dict] = []
    race_meta: dict[int, dict] = {}
    season_meta: dict[int, dict] = {}

    for n, rid in enumerate(race_ids, 1):
        race_row = valid[valid.raceId == rid].iloc[0]
        x = lap_by_race[rid]
        p = pit_by_race[rid]
        r = res_by_race.get(rid, results.iloc[0:0])
        q = qual_by_race.get(rid, qualifying.iloc[0:0])
        race_json, dmeta, rows, meta = process_race(
            rid, race_row, x, p, r, drivers, constructors, q, last_team
        )
        year = int(race_row.year)
        for driver_id, driver_meta in dmeta.items():
            driver_meta["has_img"] = has_portrait(ref_by_id.get(int(driver_id)))
        seasons[year]["year"] = year
        seasons[year]["drivers"].update(dmeta)
        seasons[year]["races"].append(race_json)
        if year not in season_meta:
            season_meta[year] = {"year": year, "races": 0, "laps": 0}
        season_meta[year]["races"] = len(seasons[year]["races"])
        season_meta[year]["laps"] += int(len(x))
        race_meta[rid] = meta
        model_records.extend(rows)
        circuit_losses[int(race_row.circuitId)].extend(meta["losses"])
        wear_all.append(
            {
                "rid": rid,
                "year": year,
                "slope": meta["wearSlope"],
                "sd": meta["wearSd"],
                "n": meta["wearN"],
                "r2": meta["wearR2"],
                "reliable": meta["wearReliable"],
                "neutralFrac": meta["neutralFrac"],
            }
        )
        if n % 40 == 0:
            print(f"  processed {n}/{len(race_ids)}  {time.time() - t0:.1f}s")

    # ---- global wear fallback ------------------------------------------------
    valid_wear = [w for w in wear_all if w["n"] >= 20 and math.isfinite(w["slope"])]
    fallback_slope = float(np.median([w["slope"] for w in valid_wear]))
    fallback_sd = float(np.median([w["sd"] for w in valid_wear if math.isfinite(w["sd"])]))
    print(
        f"wear fallback: slope {fallback_slope:.4f} s/lap, sd {fallback_sd:.3f} s, "
        f"from {len(valid_wear)} race fits; reliable {sum(1 for w in wear_all if w['reliable'])}/{len(wear_all)}"
    )

    # ---- circuit pit loss ----------------------------------------------------
    circuit_stats = []
    for cid, values in circuit_losses.items():
        c = circuits[circuits.circuitId == cid]
        name = str(c.iloc[0]["name"]) if len(c) else f"Circuit {cid}"
        country = str(c.iloc[0]["country"]) if len(c) else ""
        st = median_stats(values) or {"med": None, "p25": None, "p75": None, "sd": None, "n": 0}
        st.update({"id": int(cid), "name": name, "country": country, "ok": st["n"] >= 15})
        circuit_stats.append(st)
    circuit_stats.sort(key=lambda z: (-(z["med"] or 0), z["name"]))
    all_losses = [v for vals in circuit_losses.values() for v in vals]
    global_pit = median_stats(all_losses) or {"med": None, "p25": None, "p75": None, "sd": None, "n": 0}
    print(
        f"pit loss: {len(all_losses)} green stops, median {global_pit['med']} s; "
        f"{sum(1 for c in circuit_stats if c['ok'])}/{len(circuit_stats)} circuits have n>=15"
    )

    # ---- win model -----------------------------------------------------------
    df = pd.DataFrame(model_records)
    feat = ["f1", "f2", "f3", "f4", "f5"]
    year_by_rid = {int(k): int(v) for k, v in zip(valid.raceId, valid.year)}
    # group ids
    df = df.sort_values(["rid", "lap", "d"]).reset_index(drop=True)
    group_key = df.rid.astype(str) + "_" + df.lap.astype(str)
    codes, _ = pd.factorize(group_key, sort=False)
    df["g"] = codes
    df["year"] = df.rid.map(year_by_rid)

    train = df[df.year <= 2022].copy()
    test = df[df.year >= 2023].copy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("empty train/test split")
    mean = train[feat].mean().to_numpy(float)
    std = train[feat].std(ddof=0).to_numpy(float).copy()
    std[std == 0] = 1.0

    Xtr = (train[feat].to_numpy(float) - mean) / std
    Xte = (test[feat].to_numpy(float) - mean) / std
    gtr = train.g.to_numpy(int)
    # The test groups are a subset of the global codes; reindex them to 0..G-1.
    gte = pd.factorize(test.g.to_numpy(int))[0]
    ytr = train.y.to_numpy(int)
    yte = test.y.to_numpy(int)
    print(f"model rows: train {len(train)} ({train.g.nunique()} groups), test {len(test)} ({test.g.nunique()} groups)")

    res = fit_conditional_logit(Xtr, gtr, ytr)
    beta = res.x
    print(f"fit: {res.message}  loss {res.fun:.6f}  beta {np.round(beta, 5)}")

    pos_mean = float(train.f1.mean())
    pos_std = float(train.f1.std(ddof=0)) or 1.0
    res_pos = fit_conditional_logit(((train[["f1"]].to_numpy(float) - pos_mean) / pos_std), gtr, ytr)

    model_eval = evaluate(Xte, gte, yte, test.pos.to_numpy(int), beta)
    pos_eval = evaluate(
        ((test[["f1"]].to_numpy(float) - pos_mean) / pos_std), gte, yte, test.pos.to_numpy(int), res_pos.x
    )
    # leader baseline, smoothed at 1e-6
    groups_sorted = model_eval["groups_sorted"]
    starts = model_eval["starts"]
    sizes = model_eval["sizes"]
    y_sorted = model_eval["y_sorted"]
    pos_sorted = model_eval["pos_sorted"]
    n_repeat = np.repeat(sizes, sizes)
    eps = 1e-6
    p_leader = np.where(pos_sorted == 1, 1.0 - eps, eps / np.maximum(1, n_repeat - 1))
    p_uniform = 1.0 / n_repeat
    win_idx = model_eval["win_idx"]
    leader_logloss = -float(np.mean(np.log(np.clip(p_leader[win_idx], 1e-300, None))))
    leader_brier = float(np.mean(np.add.reduceat((p_leader - y_sorted) ** 2, starts)))
    leader_correct = 0
    leader_acc_groups = 0
    for g in range(len(starts)):
        a = starts[g]
        b = starts[g + 1] if g + 1 < len(starts) else len(p_leader)
        wi = win_idx[np.where(groups_sorted[win_idx] == g)[0][0]]
        if pos_sorted[a + (wi - a)] == 1:
            leader_correct += 1
        leader_acc_groups += 1
    leader_accuracy = leader_correct / leader_acc_groups
    uniform_logloss = -float(np.mean(np.log(np.clip(p_uniform[win_idx], 1e-300, None))))
    uniform_brier = float(np.mean(np.add.reduceat((p_uniform - y_sorted) ** 2, starts)))
    uniform_accuracy = float(np.mean(1.0 / sizes))

    calibration = calibration_bins(model_eval["p"], y_sorted, 10)

    print("\nHELD-OUT (2023-2026)")
    print(f"  model      logloss {model_eval['logloss']:.6f}  brier {model_eval['brier']:.6f}  acc {model_eval['accuracy']:.6f}")
    print(f"  position   logloss {pos_eval['logloss']:.6f}  brier {pos_eval['brier']:.6f}  acc {pos_eval['accuracy']:.6f}")
    print(f"  leader     logloss {leader_logloss:.6f}  brier {leader_brier:.6f}  acc {leader_accuracy:.6f}")
    print(f"  uniform    logloss {uniform_logloss:.6f}  brier {uniform_brier:.6f}  acc(random) {uniform_accuracy:.6f}")
    if model_eval["logloss"] >= pos_eval["logloss"]:
        raise RuntimeError("model failed to beat the position-only baseline")

    # ---- default race: clean, eventful, mid-race ----------------------------
    # Probabilities for every model row, using the fitted model.
    Xall = (df[feat].to_numpy(float) - mean) / std
    all_eval = evaluate(Xall, df.g.to_numpy(int), df.y.to_numpy(int), df.pos.to_numpy(int), beta)
    prob_by_group: dict[int, np.ndarray] = {}
    for g in range(len(all_eval["starts"])):
        a = all_eval["starts"][g]
        b = all_eval["starts"][g + 1] if g + 1 < len(all_eval["starts"]) else len(all_eval["p"])
        prob_by_group[g] = all_eval["p"][a:b]
    df["p"] = all_eval["p"]

    candidates = []
    for rid in race_ids:
        meta = race_meta[rid]
        if meta["total"] < 40:
            continue
        if meta["neutralFrac"] > 0.06:
            continue
        if meta["leadChanges"] < 4:
            continue
        mid = max(2, int(round(meta["total"] * 0.55)))
        sub = df[(df.rid == rid) & (df.lap == mid)]
        if len(sub) < 3:
            continue
        p = sub.p.to_numpy(float)
        top = float(p.max())
        contenders = int((p > 0.05).sum())
        if top > 0.85 or contenders < 3:
            continue
        score = (
            meta["leadChanges"]
            + 1.5 * contenders
            - 8.0 * meta["neutralFrac"]
            - 5.0 * abs(top - 0.55)
            + 0.02 * (int(valid.loc[valid.raceId == rid, "year"].iloc[0]) - 2011)
        )
        candidates.append((score, rid, mid, top, contenders, meta["leadChanges"], meta["neutralFrac"]))
    candidates.sort(reverse=True)
    print("\nDEFAULT RACE CANDIDATES")
    for c in candidates[:10]:
        race_row = valid[valid.raceId == c[1]].iloc[0]
        print(
            f"  {int(race_row.year)} {race_row['name']:<28} rid {c[1]} lap {c[2]} "
            f"top {c[3]:.3f} contenders {c[4]} leadChanges {c[5]} neutral {c[6] * 100:.1f}% score {c[0]:.1f}"
        )
    if not candidates:
        raise RuntimeError("no default race candidate")
    best = candidates[0]
    default_rid = int(best[1])
    default_lap = int(best[2])
    default_year = int(valid.loc[valid.raceId == default_rid, "year"].iloc[0])
    default_sub = df[(df.rid == default_rid) & (df.lap == default_lap)].sort_values("pos")
    default_focus = int(default_sub.iloc[1].d) if len(default_sub) > 1 else int(default_sub.iloc[0].d)
    print(f"default: {default_year} rid {default_rid} lap {default_lap} focus {default_focus}")

    # ---- index ---------------------------------------------------------------
    seasons_out = []
    for year in sorted(seasons):
        s = seasons[year]
        s["races"].sort(key=lambda r: r["round"])
        races_meta = []
        for race in s["races"]:
            meta = race_meta[race["id"]]
            races_meta.append(
                {
                    "id": race["id"],
                    "round": race["round"],
                    "name": race["name"],
                    "circuitId": race["circuitId"],
                    "laps": race["laps"],
                    "date": race["date"],
                    "leadChanges": meta["leadChanges"],
                    "neutralLaps": len(meta["neutral"]),
                }
            )
        seasons_out.append(
            {
                "year": year,
                "races": races_meta,
                "driverCount": len(s["drivers"]),
                "lapCount": season_meta.get(year, {}).get("laps", 0),
            }
        )
        out = {"year": year, "drivers": s["drivers"], "races": s["races"]}
        with open(OUT / f"s{year}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        print(f"wrote s{year}.json  {len(s['races'])} races  drivers {len(s['drivers'])}")

    champions = build_champions(
        races, driver_standings, constructor_standings, results, drivers, constructors, seasons_out
    )
    replay_circuit_ids = {int(v) for v in valid.circuitId.unique()}
    circuit_layouts = build_circuit_layouts(circuits, replay_circuit_ids)
    portrait_count = len(list(PORTRAIT_DIR.glob("*.png"))) if PORTRAIT_DIR.exists() else 0
    constructor_colors = all_constructor_colors(constructors)
    print(
        f"champions: {len(champions)} seasons, "
        f"{sum(1 for c in champions if c['driver'])} drivers, "
        f"{sum(1 for c in champions if c['constructor'])} constructors; portraits {portrait_count}"
    )
    print(f"constructor colors: {len(constructor_colors)} refs mapped")

    index = {
        "generated": str(date.today()),
        "source": "Ergast-style CSV archive",
        "seasons": seasons_out,
        "champions": champions,
        "constructorColors": constructor_colors,
        "circuitLayouts": circuit_layouts,
        "driverImages": {"dir": "assets/drivers", "size": 160, "count": portrait_count},
        "default": {
            "year": default_year,
            "raceId": default_rid,
            "lap": default_lap,
            "focusId": default_focus,
        },
        "circuits": circuit_stats,
        "globalPitLoss": global_pit,
        "wearFallback": {
            "slope": rnd(fallback_slope, 6),
            "sd": rnd(fallback_sd, 4),
            "nRaces": len(valid_wear),
            "note": "all-race median used when a race fit is unreliable",
        },
        "modelSummary": {
            "testLogLoss": rnd(model_eval["logloss"], 6),
            "positionLogLoss": rnd(pos_eval["logloss"], 6),
            "leaderLogLoss": rnd(leader_logloss, 6),
            "uniformLogLoss": rnd(uniform_logloss, 6),
            "testAccuracy": rnd(model_eval["accuracy"], 6),
            "positionAccuracy": rnd(pos_eval["accuracy"], 6),
            "leaderAccuracy": rnd(leader_accuracy, 6),
            "testYears": "2023-2026",
            "trainYears": "2011-2022",
        },
        "meta": {
            "raceCount": len(race_ids),
            "lapCount": int(df.shape[0]),
            "pitStopCount": int(len(all_losses)),
            "tyreCompounds": False,
            "weather": False,
            "neutralMethod": "field median > 1.08 x p15 of race field medians; lap 1 excluded",
            "pitLossMethod": "in-lap + out-lap - 2 x median of 3 clean laps before; green stops only; 5-45 s",
            "wearMethod": "OLS lap time ~ stint age + race lap, driver fixed effects (demeaned)",
            "modelTrainYears": "2011-2022",
            "modelTestYears": "2023-2026",
        },
    }
    with open(OUT / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    model_json = {
        "features": [
            "ln(position)",
            "ln(position) * laps_left/total",
            "min(gap_to_leader,300)/sqrt(laps_left+1)",
            "clip(pace_vs_leader,-3,5) * sqrt(laps_left+1)/10",
            "clip(stops_made - field_median, -2, 2)",
        ],
        "coef": [rnd(v, 12) for v in beta.tolist()],
        "scaler": {"mean": [rnd(v, 12) for v in mean.tolist()], "std": [rnd(v, 12) for v in std.tolist()]},
        "lambda": LAMBDA,
        "train": {
            "years": "2011-2022",
            "races": int(train.rid.nunique()),
            "groups": int(train.g.nunique()),
            "rows": int(len(train)),
        },
        "test": {
            "years": "2023-2026",
            "races": int(test.rid.nunique()),
            "groups": int(test.g.nunique()),
            "rows": int(len(test)),
        },
        "metrics": {
            "model": {
                "logloss": rnd(model_eval["logloss"], 9),
                "brier": rnd(model_eval["brier"], 9),
                "accuracy": rnd(model_eval["accuracy"], 9),
            },
            "position": {
                "logloss": rnd(pos_eval["logloss"], 9),
                "brier": rnd(pos_eval["brier"], 9),
                "accuracy": rnd(pos_eval["accuracy"], 9),
                "coef": rnd(float(res_pos.x[0]), 9),
                "mean": rnd(pos_mean, 9),
                "std": rnd(pos_std, 9),
            },
            "leader": {
                "logloss": rnd(leader_logloss, 9),
                "brier": rnd(leader_brier, 9),
                "accuracy": rnd(leader_accuracy, 9),
                "note": "p=1 for the current leader, clipped at 1e-6",
            },
            "uniform": {
                "logloss": rnd(uniform_logloss, 9),
                "brier": rnd(uniform_brier, 9),
                "accuracy": rnd(uniform_accuracy, 9),
                "note": "accuracy is expected random-guess accuracy, mean(1/n)",
            },
        },
        "calibration": calibration,
        "defaultSeed": RANDOM_SEED,
        "notes": [
            "Conditional logit, softmax over running cars, one winner per race-lap group.",
            "Features standardised with training mean/std; L2 1e-3; L-BFGS.",
            "Gap to leader is cumulative lap-time difference; pit lane time is already in lap times.",
            "Recent pace uses up to five clean laps; lap noise sd for the undercut is the race regression residual sd.",
        ],
    }
    with open(OUT / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=1, allow_nan=False)

    total = sum(p.stat().st_size for p in OUT.glob("*.json"))
    print(f"\nwrote index.json and model.json; data folder {total / 1024 / 1024:.2f} MB")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
