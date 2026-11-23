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

import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"

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
        if len(drow):
            dr = drow.iloc[0]
            code = fmt_driver_code(dr)
            name = f"{dr.forename} {dr.surname}".strip()
        else:
            code = "???"
            name = f"Driver {d}"
        season_driver_meta[str(d)] = {"code": code, "name": name}

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
        if team_id is not None:
            crow = constructors_df.loc[constructors_df.constructorId == team_id]
            if len(crow):
                team = str(crow.iloc[0]["name"])

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

    index = {
        "generated": str(date.today()),
        "source": "Ergast-style CSV archive",
        "seasons": seasons_out,
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
