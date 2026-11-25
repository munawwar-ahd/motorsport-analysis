"""Independent verification for PIT WALL.

Recomputes the default race/lap features and win probabilities from the raw
CSV archive, then compares them with the values the page reports through
window.PITWALL.debug(). Also spot-checks one pit loss, the Shanghai circuit
pit-loss median at race 843, and the 2011 Chinese GP tyre-wear slope with a
full driver-dummy OLS. Finally validates the champions table, original
constructor hex colours, every replay circuit layout and every local driver
portrait (including that RGB colour channels were preserved).

Run:  python verify.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
YEAR = 2011
RID = 843
LAP = 31
TOL = 1e-6
HEX_COLOR = re.compile(r"^#[0-9A-F]{6}$")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name, encoding="utf-8-sig", na_values=["\\N"])


def median_js(values: list[float]) -> float:
    v = sorted(values)
    n = len(v)
    if n == 0:
        return float("nan")
    if n % 2:
        return float(v[n // 2])
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def race_arrays(lap_times: pd.DataFrame, pits: pd.DataFrame, rid: int):
    x = lap_times[lap_times.raceId == rid].copy()
    max_lap = int(x.lap.max())
    drivers = sorted(int(d) for d in x.driverId.unique())
    idx = {d: i for i, d in enumerate(drivers)}
    ms = np.full((len(drivers), max_lap), np.nan)
    pos = np.full((len(drivers), max_lap), np.nan)
    for row in x.itertuples(index=False):
        i = idx[int(row.driverId)]
        ms[i, int(row.lap) - 1] = float(row.milliseconds)
        pos[i, int(row.lap) - 1] = float(row.position)

    inout = np.zeros((len(drivers), max_lap), dtype=bool)
    pit_laps: dict[int, list[int]] = {d: [] for d in drivers}
    p = pits[pits.raceId == rid]
    for row in p.itertuples(index=False):
        d = int(row.driverId)
        if d not in idx:
            continue
        lap = int(row.lap)
        pit_laps[d].append(lap)
        inout[idx[d], lap - 1] = True
        if lap < max_lap:
            inout[idx[d], lap] = True
    for d in pit_laps:
        pit_laps[d].sort()
    return x, max_lap, drivers, idx, ms, pos, inout, pit_laps


def neutral_laps(ms: np.ndarray, inout: np.ndarray, max_lap: int) -> set[int]:
    field_med: dict[int, float] = {}
    for lap in range(2, max_lap + 1):
        col = ms[:, lap - 1]
        mask = np.isfinite(col) & ~inout[:, lap - 1]
        if int(mask.sum()) >= 5:
            field_med[lap] = float(np.median(col[mask]))
    if not field_med:
        return set()
    p15 = float(np.percentile(np.asarray(list(field_med.values())), 15))
    threshold = 1.08 * p15
    return {lap for lap, med in field_med.items() if med > threshold}


def pace_series(ms, inout, neutral, max_lap):
    n = ms.shape[0]
    pace = np.full((n, max_lap), np.nan)
    for i in range(n):
        for lap in range(1, max_lap + 1):
            vals: list[float] = []
            for ll in range(lap, max(0, lap - 10), -1):
                if ll < 1:
                    break
                if (
                    ll >= 2
                    and ll not in neutral
                    and not inout[i, ll - 1]
                    and np.isfinite(ms[i, ll - 1])
                ):
                    vals.append(float(ms[i, ll - 1]))
                if len(vals) == 5:
                    break
            if len(vals) < 2:
                vals = []
                for ll in range(lap, 0, -1):
                    if np.isfinite(ms[i, ll - 1]):
                        vals.append(float(ms[i, ll - 1]))
                    if len(vals) == 5:
                        break
            if vals:
                pace[i, lap - 1] = median_js(vals)
    return pace


def tyre_age(pit_laps: list[int], last_lap: int, lap: int) -> int | None:
    starts = [1] + [p + 1 for p in pit_laps if 1 <= p < last_lap]
    ends = [p for p in pit_laps if 1 <= p < last_lap] + [last_lap]
    for a, b in zip(starts, ends):
        if a <= lap <= b:
            return lap - a + 1
    return None


def pit_loss_values(lap_times, pits, races, rid):
    """Recompute every valid green stop in one race, using raw lap times."""
    _, max_lap, drivers, idx, ms, _, inout, pit_laps = race_arrays(lap_times, pits, rid)
    neutral = neutral_laps(ms, inout, max_lap)
    out = []
    for d in drivers:
        i = idx[d]
        for lap in pit_laps[d]:
            if lap < 2 or lap >= max_lap or lap in neutral or (lap + 1) in neutral:
                continue
            in_ms, out_ms = ms[i, lap - 1], ms[i, lap]
            if not (np.isfinite(in_ms) and np.isfinite(out_ms)):
                continue
            before = []
            for ll in range(lap - 1, 1, -1):
                if ll in neutral or inout[i, ll - 1]:
                    continue
                if np.isfinite(ms[i, ll - 1]):
                    before.append(float(ms[i, ll - 1]))
                if len(before) == 3:
                    break
            if len(before) < 3:
                continue
            loss = (float(in_ms) + float(out_ms) - 2 * float(np.median(before))) / 1000.0
            if 5 < loss < 45:
                out.append((d, lap, loss))
    return out


def circuit_loss_pool(lap_times, pits, races, circuit_id):
    values = []
    for rid in sorted(lap_times.raceId.unique()):
        if rid not in set(pits.raceId):
            continue
        r = races.loc[races.raceId == rid]
        if len(r) == 0 or int(r.iloc[0].circuitId) != circuit_id:
            continue
        values.extend(v for _, _, v in pit_loss_values(lap_times, pits, races, rid))
    return np.asarray(values, dtype=float)


def tyre_slope(lap_times, pits, rid):
    _, max_lap, drivers, idx, ms, _, inout, pit_laps = race_arrays(lap_times, pits, rid)
    neutral = neutral_laps(ms, inout, max_lap)
    rows = []
    for d in drivers:
        i = idx[d]
        starts = [1] + [p + 1 for p in pit_laps[d] if 1 <= p < max_lap]
        ends = [p for p in pit_laps[d] if 1 <= p < max_lap] + [max_lap]
        age = np.zeros(max_lap)
        for a, b in zip(starts, ends):
            if 1 <= a <= b <= max_lap:
                age[a - 1 : b] = np.arange(1, b - a + 2)
        for lap in range(2, max_lap + 1):
            if not np.isfinite(ms[i, lap - 1]) or lap in neutral or inout[i, lap - 1]:
                continue
            rows.append((d, float(age[lap - 1]), lap, float(ms[i, lap - 1])))
    df = pd.DataFrame(rows, columns=["d", "age", "lap", "ms"])
    # Full driver-dummy OLS: this is the independent check on the within fit.
    dummies = pd.get_dummies(df["d"], drop_first=True).to_numpy(float)
    X = np.column_stack([df.age.to_numpy(float), df.lap.to_numpy(float), dummies, np.ones(len(df))])
    y = df.ms.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2_full = 1.0 - float(resid @ resid) / float((y - y.mean()) @ (y - y.mean()))
    # The page reports within R2 (driver means removed).
    dm = df.copy()
    for col in ("age", "lap", "ms"):
        dm[col] = df[col] - df.groupby("d")[col].transform("mean")
    b2 = np.linalg.lstsq(
        np.column_stack([dm.age.to_numpy(float), dm.lap.to_numpy(float)]),
        dm.ms.to_numpy(float),
        rcond=None,
    )[0]
    resid2 = dm.ms.to_numpy(float) - np.column_stack(
        [dm.age.to_numpy(float), dm.lap.to_numpy(float)]
    ) @ b2
    sst2 = float((dm.ms.to_numpy(float) ** 2).sum())
    r2_within = 1.0 - float(resid2 @ resid2) / sst2
    return {
        "n": len(df),
        "slope": float(beta[0] / 1000.0),
        "slope_within": float(b2[0] / 1000.0),
        "r2_full": r2_full,
        "r2_within": r2_within,
    }


def verify_champions_layouts_portraits(index: dict) -> None:
    """Independent checks for the three new offline assets in index.json."""
    races_all = read_csv("races.csv")
    drivers_all = read_csv("drivers.csv")
    years = sorted(int(y) for y in races_all.year.unique())
    champions = index.get("champions", [])
    by_year = {int(c["year"]): c for c in champions}
    missing = [y for y in years if y not in by_year]
    assert not missing, f"champions missing seasons: {missing}"
    bad = []
    constructor_refs = {str(r.constructorRef) for r in constructors_all.itertuples(index=False)}
    color_map = index.get("constructorColors") or {}
    if set(color_map) != constructor_refs:
        bad.append("constructorColors does not cover every constructors.csv ref")
    for ref, color in color_map.items():
        if not isinstance(color, str) or not HEX_COLOR.fullmatch(color):
            bad.append(f"constructorColors[{ref}]: invalid hex {color!r}")
    replay_years = {int(s["year"]): s for s in index["seasons"]}
    for year, row in by_year.items():
        d = row.get("driver") or {}
        con = row.get("constructor") or {}
        if year not in years:
            bad.append(f"{year}: unknown season")
        if not all(d.get(k) is not None for k in ("driverId", "driverRef", "code", "name", "points", "wins", "podiums", "margin")):
            bad.append(f"{year}: incomplete champion driver")
        if year >= 1958 and not all(con.get(k) is not None for k in ("constructorId", "name", "points", "wins")):
            bad.append(f"{year}: incomplete champion constructor")
        if year < 1958 and con:
            bad.append(f"{year}: constructor champion before 1958")
        if row.get("has_replay") != (year in replay_years):
            bad.append(f"{year}: has_replay mismatch")
        if year in replay_years:
            race_ids = {int(r["id"]) for r in replay_years[year]["races"]}
            if row.get("default_race_id") not in race_ids:
                bad.append(f"{year}: default_race_id not in replay set")
    assert not bad, "champion records / colours: " + "; ".join(bad[:12])
    expected = {
        "ferrari": "#E8002D", "red_bull": "#3671C6", "mercedes": "#27F4D2",
        "mclaren": "#FF8000", "aston_martin": "#229971", "alpine": "#00A1E8",
        "williams": "#1868DB", "rb": "#6692FF", "alphatauri": "#2B4562",
        "toro_rosso": "#469BFF", "sauber": "#52E252", "alfa": "#C92D4B",
        "haas": "#B6BABD", "audi": "#F50537", "cadillac": "#909090",
        "renault": "#FFF500", "racing_point": "#F596C8", "force_india": "#FF5F0F",
        "team_lotus": "#FFB800", "brawn": "#B8FD6E", "jordan": "#EED202",
        "benetton": "#00A86B", "brabham": "#002855", "tyrrell": "#003399",
        "matra": "#0055A4", "cooper": "#004225", "brm": "#004225",
        "vanwall": "#006633", "maserati": "#C8102E",
    }
    wrong = [f"{ref}: {color_map.get(ref)} != {hex}" for ref, hex in expected.items() if color_map.get(ref) != hex]
    assert not wrong, "original constructor colours: " + "; ".join(wrong[:12])
    print(f"\nChampions: {len(champions)} seasons ({years[0]}-{years[-1]}), "
          f"{sum(1 for c in champions if c.get('constructor'))} constructor champions, "
          f"{sum(1 for c in champions if c.get('has_replay'))} with replay data")
    print(f"Constructor colours: {len(color_map)} refs mapped, "
          f"{sum(1 for c in champions if c.get('constructor'))} champion badges coloured")

    layouts = index.get("circuitLayouts", {})
    replay_circuits = set()
    for season in index["seasons"]:
        for race in season["races"]:
            replay_circuits.add(int(race["circuitId"]))
    missing_layouts = []
    point_counts = []
    era_count = 0
    for cid in sorted(replay_circuits):
        entry = layouts.get(str(cid))
        if not entry:
            missing_layouts.append(f"{cid}: no entry")
            continue
        path = entry.get("svg_path") or ""
        if not entry.get("fallback") and (not path.startswith("M") or not path.endswith("Z") or len(path) < 200):
            missing_layouts.append(f"{cid}: empty svg_path")
            continue
        if not entry.get("fallback") and len(entry.get("sf_line") or []) != 4:
            missing_layouts.append(f"{cid}: missing sf_line")
        point_counts.append(len(path.split(" L")) if path else 0)
        era_count += len(entry.get("eras") or [])
        for era in entry.get("eras") or []:
            if era.get("from", 9999) > era.get("to", 0):
                missing_layouts.append(f"{cid}: invalid era range")
    assert not missing_layouts, "circuit layouts: " + "; ".join(missing_layouts[:12])
    print(f"Circuit layouts: {len(replay_circuits)} replay circuits, "
          f"{sum(1 for c in replay_circuits if not layouts[str(c)]['fallback'])} with geometry, "
          f"{era_count} era-specific variants; {min(point_counts)}-{max(point_counts)} points per track")

    ref_of = {int(r.driverId): str(r.driverRef) for r in drivers_all.itertuples(index=False)}
    refs = set()
    for season in index["seasons"]:
        data = json.loads((ROOT / "data" / f"s{season['year']}.json").read_text(encoding="utf-8"))
        for driver_id, meta in data["drivers"].items():
            if meta.get("has_img"):
                refs.add(ref_of[int(driver_id)])
    for row in champions:
        d = row.get("driver") or {}
        if d.get("has_img"):
            refs.add(str(d["driverRef"]))
    broken = []
    colorful = 0
    for ref in sorted(refs):
        path = ROOT / "assets" / "drivers" / f"{ref}.png"
        if not path.is_file():
            broken.append(f"{ref}: missing")
            continue
        try:
            with Image.open(path) as img:
                if img.format != "PNG" or img.size != (160, 160) or img.mode != "RGBA":
                    broken.append(f"{ref}: {img.format} {img.size} {img.mode}")
                arr = np.asarray(img.convert("RGBA"))
                rgb = arr[..., :3].astype(np.int16)
                spread = rgb.max(axis=2) - rgb.min(axis=2)
                if np.any((spread > 12) & (arr[..., 3] > 0)):
                    colorful += 1
        except Exception as exc:
            broken.append(f"{ref}: {exc}")
    assert not broken, "portraits: " + "; ".join(broken[:12])
    assert colorful > len(refs) / 2, f"portraits appear greyscale: only {colorful}/{len(refs)} have colour pixels"
    on_disk = len(list((ROOT / "assets" / "drivers").glob("*.png")))
    assert on_disk == index.get("driverImages", {}).get("count"), "driverImages.count mismatch"
    print(f"Driver portraits: {len(refs)} referenced refs valid 160x160 RGBA PNG; {on_disk} files on disk")
    print(f"Portrait colour: {colorful}/{len(refs)} contain real RGB colour pixels (no greyscale conversion)")
    print("PASS: champions, original team colours, layouts and full-colour portraits are valid")


def main():
    lap_times = read_csv("lap_times.csv")
    pits = read_csv("pit_stops.csv")
    races = read_csv("races.csv")
    # Match the pipeline scope: 2011+ races that have both lap times and stops.
    valid = set(int(r) for r in races.loc[races.year >= 2011, "raceId"])
    valid &= set(int(r) for r in lap_times.raceId.unique())
    valid &= set(int(r) for r in pits.raceId.unique())
    lap_times = lap_times[lap_times.raceId.isin(valid)].copy()
    pits = pits[pits.raceId.isin(valid)].copy()
    races = races[races.raceId.isin(valid)].copy()
    model = json.loads((ROOT / "data/model.json").read_text(encoding="utf-8"))
    page = json.loads((ROOT / "page_debug.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "data/index.json").read_text(encoding="utf-8"))

    assert page["raceId"] == RID and page["lap"] == LAP, "page debug is not the default race/lap"

    _, max_lap, drivers, idx, ms, pos, inout, pit_laps = race_arrays(lap_times, pits, RID)
    neutral = neutral_laps(ms, inout, max_lap)
    pace = pace_series(ms, inout, neutral, max_lap)
    cum = np.nancumsum(ms, axis=1)
    running = [d for d in drivers if np.isfinite(ms[idx[d], LAP - 1])]
    leader = min(running, key=lambda d: pos[idx[d], LAP - 1])
    stops = {d: sum(1 for p in pit_laps[d] if p < LAP) for d in drivers}
    med_stops = median_js([float(stops[d]) for d in running])

    order = sorted(running, key=lambda d: (pos[idx[d], LAP - 1], d))
    gap = {d: max(0.0, (cum[idx[d], LAP - 1] - cum[idx[leader], LAP - 1]) / 1000.0) for d in running}
    interval = {}
    for i, d in enumerate(order):
        interval[d] = None if i == 0 else max(
            0.0, (cum[idx[d], LAP - 1] - cum[idx[order[i - 1]], LAP - 1]) / 1000.0
        )

    coef = np.asarray(model["coef"], float)
    mean = np.asarray(model["scaler"]["mean"], float)
    std = np.asarray(model["scaler"]["std"], float)
    laps_left = max_lap - LAP
    pace_lead = pace[idx[leader], LAP - 1]
    features = {}
    for d in running:
        p = float(pos[idx[d], LAP - 1])
        pdiff = 0.0 if not np.isfinite(pace[idx[d], LAP - 1]) or not np.isfinite(pace_lead) else float(
            np.clip((pace[idx[d], LAP - 1] - pace_lead) / 1000.0, -3, 5)
        )
        f = [
            math.log(p),
            math.log(p) * laps_left / max_lap,
            min(gap[d], 300.0) / math.sqrt(laps_left + 1),
            pdiff * math.sqrt(laps_left + 1) / 10.0,
            float(np.clip(stops[d] - med_stops, -2, 2)),
        ]
        features[d] = f
    scores = np.asarray([(np.asarray(features[d]) - mean) / std @ coef for d in order])
    ex = np.exp(scores - scores.max())
    probs = ex / ex.sum()

    code_of = {}
    s2011 = json.loads((ROOT / "data/s2011.json").read_text(encoding="utf-8"))
    for season in [s2011]:
        for k, v in season["drivers"].items():
            code_of[int(k)] = v["code"]

    worst = {"gap": (0.0, None), "prob": (0.0, None)}
    print(f"Independent recomputation, {YEAR} race {RID}, lap {LAP}, {len(running)} running")
    print(f"neutral laps: {len(neutral)}; median stops: {med_stops:g}")
    print("code      pos   gap(s)      page gap    diff        prob          page prob     diff")
    for d, p in zip(order, probs):
        c = code_of[d]
        pg = page["gaps"][c]
        pp = page["probs"][c]
        dg = abs(gap[d] - pg)
        dp = abs(float(p) - pp)
        if dg > worst["gap"][0]:
            worst["gap"] = (dg, c)
        if dp > worst["prob"][0]:
            worst["prob"] = (dp, c)
        print(f"{c:>3} {int(pos[idx[d], LAP - 1]):5d} {gap[d]:11.6f} {pg:11.6f} {dg:10.3e} {p:13.9f} {pp:13.9f} {dp:10.3e}")
        if interval[d] is None:
            assert page["intervals"][c] is None, f"interval mismatch for {c}"
        else:
            assert page["intervals"][c] is not None and abs(interval[d] - page["intervals"][c]) < TOL, f"interval mismatch for {c}"
        assert abs(float(pos[idx[d], LAP - 1]) - page["positions"][c]) < TOL
        age = tyre_age(pit_laps[d], int(np.isfinite(ms[idx[d]]).sum()), LAP)
        assert (age is None and page["tyreAge"][c] is None) or age == page["tyreAge"][c], f"tyre age mismatch for {c}"
        assert stops[d] == page["stops"][c], f"stop count mismatch for {c}"

    assert abs(float(probs.sum()) - 1.0) < 1e-12
    print(f"worst gap error:  {worst['gap'][0]:.3e} s ({worst['gap'][1]})")
    print(f"worst prob error: {worst['prob'][0]:.3e} ({worst['prob'][1]})")
    assert worst["gap"][0] < TOL and worst["prob"][0] < TOL
    print("PASS: page gaps and probabilities match raw-CSV recomputation to < 1e-6")

    # ---- spot check: one pit loss, by hand --------------------------------
    losses = pit_loss_values(lap_times, pits, races, RID)
    vet = [v for d, lap, v in losses if d == 20 and lap == 14][0]
    x = lap_times[(lap_times.raceId == RID) & (lap_times.driverId == 20)].set_index("lap")
    before = [x.loc[ll, "milliseconds"] for ll in (13, 12, 11)]
    manual = (x.loc[14, "milliseconds"] + x.loc[15, "milliseconds"] - 2 * np.median(before)) / 1000
    print("\nPit-loss spot check: VET, 2011 Chinese GP, stop on lap 14")
    print(f"  in-lap {x.loc[14, 'milliseconds'] / 1000:.3f} s + out-lap {x.loc[15, 'milliseconds'] / 1000:.3f} s"
          f" - 2 x median(11: {before[2] / 1000:.3f}, 12: {before[1] / 1000:.3f}, 13: {before[0] / 1000:.3f})"
          f" = {manual:.3f} s")
    assert abs(vet - manual) < 1e-9
    print(f"  pipeline value {vet:.3f} s; race median {np.median([v for _, _, v in losses]):.3f} s")

    # Displayed circuit figure: Shanghai (circuit 17), all 2011+ green stops.
    circuit_id = int(races.loc[races.raceId == RID, "circuitId"].iloc[0])
    pool = circuit_loss_pool(lap_times, pits, races, circuit_id)
    sh = next(c for c in index["circuits"] if c["id"] == circuit_id)
    print(f"Displayed circuit figure: Shanghai median {np.median(pool):.3f} s from {len(pool)} stops;"
          f" data/index.json has {sh['med']:.3f} s, n={sh['n']}")
    assert abs(np.median(pool) - sh["med"]) < 0.001 and len(pool) == sh["n"]

    # ---- spot check: tyre slope with full driver dummies -------------------
    fit = tyre_slope(lap_times, pits, RID)
    wear = json.loads((ROOT / "data/s2011.json").read_text(encoding="utf-8"))["races"]
    race_wear = next(r["wear"] for r in wear if r["id"] == RID)
    print("\nTyre-slope spot check: 2011 Chinese GP")
    print(f"  full driver-dummy OLS slope {fit['slope']:.6f} s/lap, within slope {fit['slope_within']:.6f}, n={fit['n']}")
    print(f"  page/data slope {race_wear['slope']:.6f} s/lap, within R2 {race_wear['r2']:.5f} (full R2 {fit['r2_full']:.5f})")
    assert abs(fit["slope"] - race_wear["slope"]) < 1e-6
    assert abs(fit["slope_within"] - race_wear["slope"]) < 1e-6
    assert fit["n"] == race_wear["n"]
    assert abs(fit["r2_within"] - race_wear["r2"]) < 1e-5

    # ---- held-out metrics --------------------------------------------------
    m = model["metrics"]
    print("\nHeld-out metrics (2023-2026):")
    for name in ("model", "position", "leader", "uniform"):
        row = m[name]
        print(f"  {name:9s} log-loss {row['logloss']:.6f}  Brier {row['brier']:.6f}  accuracy {row['accuracy'] * 100:.4f}%")
    assert m["model"]["logloss"] < m["position"]["logloss"]
    print("PASS: full model beats the position-only baseline on held-out log-loss")

    verify_champions_layouts_portraits(index)

    size = sum(p.stat().st_size for p in (ROOT / "data").glob("*.json"))
    print(f"\nData folder: {size / 1024 / 1024:.2f} MiB in {len(list((ROOT / 'data').glob('*.json')))} JSON files")


if __name__ == "__main__":
    main()
