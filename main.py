"""GTA6 trophy rarity forecast.

For each case file, we read the sequence of web-page refreshes (history
states), model how each trophy's earn rate evolves on a logit scale using
damped integrals over the horizon, blend a rate-drift model with a
earner-count model, shrink per-trophy drift toward tier/class means, and
emit five quantiles per trophy. Structural constraints (story chain
monotonicity, platinum cap) are applied afterward. Any failure falls back
to a plain per-trophy logit normal forecast around the last observed rate.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        e = math.exp(-l)
        return 100.0 / (1.0 + e)
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    r = math.floor(x * 10000.0 + 0.5) / 10000.0 if x >= 0 else -math.floor(-x * 10000.0 + 0.5) / 10000.0
    if r == int(r):
        return int(r)
    return r


_Z_VALUES = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}

_A = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
      138.3577518672690, -30.66479806614716, 2.506628277459239]
_B = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
      66.80131188771972, -13.28068155288572]
_C = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
      -2.549732539343734, 4.374664141464968, 2.938163982698783]
_D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
      3.754408661907416]


def z_for_level(tau):
    if tau in _Z_VALUES:
        return _Z_VALUES[tau]
    p_low = 0.02425
    if tau < p_low:
        q = math.sqrt(-2.0 * math.log(tau))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return num / den
    if tau <= 1.0 - p_low:
        q = tau - 0.5
        r = q * q
        num = ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
        num = num * q
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - tau))
    num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
    den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    return -num / den


def parse_time(s):
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def eff(H, HL):
    return (HL / math.log(2.0)) * (1.0 - 2.0 ** (-H / HL))


def lsq_fit(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def age_days(t, release):
    d = (t - release).total_seconds() / 86400.0
    return max(d, 0.01)


def build_states(history):
    states = []
    prev_players = None
    prev_rates = None
    for read in history or []:
        rates = read.get("rates")
        players = read.get("players_tracked")
        try:
            slot = parse_time(read.get("slot"))
            at = parse_time(read.get("at"))
        except Exception:
            slot = at = None
        t = slot or at
        if t is None:
            continue
        if not states or (prev_players != players) or (rates != prev_rates):
            states.append({"t": t, "players": players, "rates": rates})
            prev_players = players
            prev_rates = rates
    return states


def release_time(case, states):
    rd = (case.get("game") or {}).get("release_date")
    if isinstance(rd, str):
        try:
            dt = datetime.strptime(rd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    if states:
        return states[0]["t"]
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def target_time(case, states):
    t = parse_time(case.get("target"))
    if t is not None:
        return t
    at = parse_time(case.get("at"))
    if at is not None:
        h = case.get("horizon_minutes")
        try:
            return at + timedelta(minutes=float(h))
        except (TypeError, ValueError):
            return at + timedelta(days=1.0)
    if states:
        return states[-1]["t"] + timedelta(days=1.0)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def horizon_days(case, tt, states):
    try:
        h = case.get("horizon_minutes")
        if h is not None:
            return float(h) / 1440.0
    except (TypeError, ValueError):
        pass
    at = parse_time(case.get("at"))
    if at is not None and tt is not None:
        try:
            return max((tt - at).total_seconds() / 86400.0, 1e-6)
        except Exception:
            pass
    return 1.0


def last_rate_per_trophy(case, reads):
    n = len(case.get("trophies") or [])
    last = [None] * n
    for read in reversed(reads or []):
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        for i in range(min(n, len(rates))):
            if last[i] is None and isinstance(rates[i], (int, float)):
                last[i] = float(rates[i])
        if all(v is not None for v in last):
            break
    return [0.01 if v is None else v for v in last]


def fallback_rows(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [z_for_level(t) for t in levels]
    H = 1.0
    tt = target_time(case, build_states(case.get("history")) if case.get("history") else [])
    at = parse_time(case.get("at"))
    if at is not None:
        try:
            h = case.get("horizon_minutes")
            if h is not None:
                H = float(h) / 1440.0
            else:
                H = max((tt - at).total_seconds() / 86400.0, 1e-6)
        except (TypeError, ValueError):
            pass
    last = last_rate_per_trophy(case, case.get("history"))
    spread = 0.5 * math.sqrt(H)
    rows = []
    for r in last:
        med = logit(r)
        row = []
        prev = 0.0
        for z in zs:
            v = min(100.0, max(0.0, inv_logit(med + z * spread)))
            v = round4(max(v, prev))
            row.append(v)
            prev = v
        rows.append(row)
    return rows


def forecast_players(states, tt, H, release):
    pts = [(age_days(s["t"], release), float(s["players"]))
           for s in states if isinstance(s.get("players"), (int, float)) and s["players"] > 0]
    if not pts:
        return None
    last_p = pts[-1][1]

    x_last = math.log(1.0 + age_days(states[-1]["t"], release))
    x_target = math.log(1.0 + age_days(tt, release))
    dx = x_target - x_last

    p_a = None
    fit_pts = [(a, p) for a, p in pts if a >= 0.5]
    xs = [math.log(1.0 + a) for a, _ in fit_pts]
    if len(set(xs)) < 2:
        fit_pts = pts
        xs = [math.log(1.0 + a) for a, _ in fit_pts]
    fit = lsq_fit(xs, [math.log(p) for _, p in fit_pts])
    if fit is not None:
        a_int, b_slope = fit
        p_a = math.exp(a_int + b_slope * x_target)

    cutoff = states[-1]["t"] - timedelta(hours=48)
    recent = [s for s in states
              if isinstance(s.get("players"), (int, float)) and s["players"] > 0
              and s["t"] >= cutoff]
    if len(recent) < 2:
        recent = [s for s in states
                  if isinstance(s.get("players"), (int, float)) and s["players"] > 0]
    p_b = None
    fit_b = lsq_fit(
        [math.log(1.0 + age_days(s["t"], release)) for s in recent],
        [math.log(float(s["players"])) for s in recent],
    )
    if fit_b is not None:
        b_B = min(3.0, max(-3.0, fit_b[1]))
        p_b = last_p * math.exp(b_B * dx)

    cands = [p for p in (p_a, p_b) if p is not None and p > 0]
    if not cands:
        return None
    p_fc = math.exp(sum(math.log(c) for c in cands) / len(cands))
    p_fc = min(12.0 * last_p, max(0.9 * last_p, p_fc))
    return p_fc


def story_chain_order(trophies):
    order = []
    chain = []
    gone = None
    final = None
    chapters = []
    pat = re.compile(r"^Complete Chapter (\d+)\.$")
    for i, t in enumerate(trophies):
        d = t.get("description") or ""
        if d == "Complete Gone Fishing.":
            gone = i
        elif d == "Complete the final mission.":
            final = i
        else:
            m = pat.match(d)
            if m:
                chapters.append((int(m.group(1)), i))
    if gone is not None:
        chain.append(gone)
    chain.extend(i for _, i in sorted(chapters))
    if final is not None:
        chain.append(final)
    return chain


def finish_row(row):
    out = []
    prev = 0.0
    for v in row:
        v = min(100.0, max(0.0, v))
        v = round4(max(v, prev))
        out.append(v)
        prev = v
    return out


def main_forecast(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [z_for_level(t) for t in levels]
    history = case.get("history") or []
    states = build_states(history)
    if len(states) < 2:
        return None
    last_p = states[-1].get("players")
    if not isinstance(last_p, (int, float)) or last_p <= 0:
        return None

    release = release_time(case, states)
    tt = target_time(case, states)
    H = horizon_days(case, tt, states)
    P_fc = forecast_players(states, tt, H, release)
    if P_fc is None or P_fc <= 0:
        return None

    trophies = case.get("trophies") or []
    n = len(trophies)

    obs_all = []
    for i in range(n):
        obs = []
        for s in states:
            rates = s.get("rates")
            if isinstance(rates, list) and i < len(rates) and isinstance(rates[i], (int, float)):
                pl = s.get("players")
                e = float(rates[i]) / 100.0 * float(pl) if isinstance(pl, (int, float)) else 0.0
                obs.append((age_days(s["t"], release), float(rates[i]), e, s["t"]))
        obs_all.append(obs)

    last_state_t = states[-1]["t"]
    cutoff = last_state_t - timedelta(hours=48)
    dx_fc = math.log(1.0 + age_days(tt, release)) - math.log(1.0 + age_days(last_state_t, release))

    per = []
    for i, t in enumerate(trophies):
        obs = obs_all[i]
        if not obs:
            per.append(None)
            continue
        r_last = obs[-1][1]
        e_last = obs[-1][2]

        W = [o for o in obs if o[3] >= cutoff]
        if len(W) < 4:
            W = obs[-4:]
        if len(W) < 2:
            W = obs

        xs_w = [math.log(1.0 + o[0]) for o in W]
        fit_b = lsq_fit(xs_w, [logit(o[1]) for o in W])
        b_i = 0.0 if fit_b is None else min(6.0, max(-6.0, fit_b[1]))

        fit_c = lsq_fit(xs_w, [math.log(max(o[2], 0.25)) for o in W])
        c_i = 0.0 if fit_c is None else min(6.0, max(-3.0, fit_c[1]))
        if e_last == 0 and all(o[2] == 0 for o in W):
            c_i = 0.0

        m_i = sum(1 for a, b in zip(W, W[1:]) if a[1] != b[1])

        per.append({
            "r_last": r_last, "e_last": e_last, "b": b_i, "c": c_i, "m": m_i,
            "tier": t.get("tier"),
            "obs": obs,
        })

    # Class means by tier for drift and g.
    def weighted_mean(items):
        num = 0.0
        den = 0.0
        for p in items:
            w = min(p["e_last"], 50.0) + 1.0
            num += w * p["_val"]
            den += w
        return num / den if den > 0 else None

    tiers = {}
    for i, p in enumerate(per):
        if p is None:
            continue
        tiers.setdefault(p["tier"], []).append(p)

    for p in per:
        if p is None:
            continue
        members = tiers.get(p["tier"], [])
        if members:
            for key, out in (("b", "_cd"), ("c", "_cg")):
                for q in members:
                    q["_val"] = q[key]
                wm = weighted_mean(members)
                p[out] = wm if wm is not None else 0.0
        else:
            p["_cd"] = p["_cg"] = 0.0
    allp = [p for p in per if p is not None]
    if allp:
        for key, out in (("b", "_gd"), ("c", "_gg")):
            for q in allp:
                q["_val"] = q[key]
            wm = weighted_mean(allp)
            gm = wm if wm is not None else 0.0
            for p in per:
                if p is not None and p.get(out) is None:
                    p[out] = gm
    for p in per:
        if p is not None:
            if p.get("_cd") is None:
                p["_cd"] = p.get("_gd", 0.0)
            if p.get("_cg") is None:
                p["_cg"] = p.get("_gg", 0.0)

    rows = []
    for i, t in enumerate(trophies):
        p = per[i]
        if p is None:
            r = last_rate_per_trophy(case, history)[i]
            row = [inv_logit(logit(r) + z * 0.5 * math.sqrt(H)) for z in zs]
            rows.append(finish_row(row))
            continue

        r_last = p["r_last"]
        e_last = p["e_last"]
        w = p["m"] / (p["m"] + 3.0)
        b_shrunk = w * p["b"] + (1 - w) * p["_cd"]
        c_shrunk = w * p["c"] + (1 - w) * p["_cg"]

        L1 = logit(r_last) + b_shrunk * dx_fc

        if e_last > 0:
            e_fc = e_last * math.exp(c_shrunk * dx_fc)
            e_fc = max(e_fc, 0.85 * e_last)
            e_fc = min(e_fc, 0.995 * P_fc, e_last * math.exp(3.0))
            rate2 = 100.0 * e_fc / P_fc
            L2 = logit(rate2) if e_fc > 0 else None
        else:
            L2 = None

        if r_last >= 99.9:
            logit_med = logit(99.9)
            sigma = 0.22
        else:
            avail = [L1] + ([L2] if L2 is not None else [])
            logit_med = sum(avail) / len(avail)
            logit_med = min(logit(r_last) + 2.6, max(logit(r_last) - 2.6, logit_med))
            logit_med = logit(r_last) + 0.45 * (logit_med - logit(r_last))
            move_i = logit_med - logit(r_last)
            sigma = math.sqrt(
                (0.35 * math.sqrt(H)) ** 2
                + (0.5 * abs(move_i)) ** 2
                + 0.30 ** 2 * min(1.0, 15.0 / max(e_last, 1.0))
            )
            sigma = min(1.80, max(0.35, sigma))

        row = [inv_logit(logit_med + z * sigma) for z in zs]
        rows.append(finish_row(row))

    # Structure clamps.
    for k in range(len(levels)):
        run = 100.0
        for i in story_chain_order(trophies):
            run = min(run, rows[i][k])
            rows[i][k] = run
        plat = next((i for i, t in enumerate(trophies) if t.get("tier") == "platinum"), None)
        if plat is not None and n > 1:
            m = min(rows[j][k] for j in range(n) if j != plat)
            if rows[plat][k] > m:
                rows[plat][k] = m

    return rows


def write_case(case, out_dir, stem):
    cid = stem or case.get("id") or "case"
    try:
        rows = main_forecast(case)
        if rows is None:
            rows = fallback_rows(case)
    except Exception:
        try:
            rows = fallback_rows(case)
        except Exception:
            n = len(case.get("trophies") or [])
            rows = [[50.0] * 5 for _ in range(n)]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    levels = levels[:len(rows[0])] if rows else levels
    trophies = case.get("trophies") or []
    out = {"trophies": [
        {"id": t.get("id"), "q": rows[i] if i < len(rows) else [50.0] * len(levels)}
        for i, t in enumerate(trophies)
    ]}
    out_path = Path(out_dir) / f"{cid}.out"
    out_path.write_text(json.dumps(out, separators=(",", ":")) + "\n")


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(in_dir).glob("*.in")):
        try:
            case = json.loads(path.read_text())
        except Exception:
            continue
        try:
            write_case(case, out_dir, path.stem)
        except Exception:
            pass


if __name__ == "__main__":
    main()
