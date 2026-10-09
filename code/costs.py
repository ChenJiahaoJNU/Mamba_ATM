"""A cost model with the parts that actually differ between the two sides of an A-share trade.

The earlier evaluations charged one symmetric rate per position change. That is too generous in some
places and too harsh in others, and it says nothing about capacity. This module charges, per day and
per leg:

* commission and transfer fee on both sides,
* stamp duty on the sell side only, at 10 bp before 2023-08-28 and 5 bp after, so the rate changes
  inside the sample,
* slippage as a fixed half-spread on both sides,
* square-root market impact, ``coef * sigma_daily * sqrt(traded notional / daily turnover value)``,
  which depends on the size of the book and on the liquidity of the name being traded,
* a borrow fee on the short leg, charged per calendar year on its notional, because a long-short
  book in China cannot be assumed to borrow for free.

Impact uses information available when the signal is formed: the turnover value of the signal day and
the volatility of the look-back window. Participation above the reported threshold is flagged rather
than quietly extrapolated, because a square-root law is not a promise at a third of a day's volume.

Long-only portfolios are evaluated by the same function: no borrow fee, no short leg, stamp duty on
the sells only.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostModel:
    commission: float = 3e-4          # per side
    transfer: float = 1e-5            # per side
    slippage: float = 2e-4            # per side, half the quoted spread
    stamp_before: float = 1e-3        # sell side, before 2023-08-28
    stamp_after: float = 5e-4         # sell side, from 2023-08-28
    stamp_change_day: int = 19597     # 2023-08-28 as days since the epoch
    borrow_annual: float = 0.08       # on the short leg notional per year
    impact_coef: float = 0.5          # square-root law coefficient
    aum: float = 1e8                  # notional of one leg
    participation_warn: float = 0.10  # flag trades above this share of daily turnover value

    def stamp(self, day):
        return self.stamp_before if day < self.stamp_change_day else self.stamp_after

    def one_way(self, day, selling):
        c = self.commission + self.transfer + self.slippage
        return c + (self.stamp(day) if selling else 0.0)


def liquidity_arrays(data, ds):
    """Turnover value and return volatility of the signal day, per dataset row."""
    from xsection import FEATURE_COLS
    amount = FEATURE_COLS.index("成交额")
    adv = np.ones(len(ds.y), dtype=np.float64)
    sigma = np.full(len(ds.y), 0.02, dtype=np.float64)
    L = ds.X.shape[1]
    for k, code in enumerate(ds.codes):
        _, feat, ret = data[code]
        sel = ds.stock_idx == k
        idx = ds.end_idx[sel]
        adv[sel] = np.maximum(feat[idx, amount], 1.0)
        sigma[sel] = np.array([max(float(np.std(ret[t - (L - 1):t + 1])), 1e-4) for t in idx])
    return adv, sigma


def replay(scores, y, di, grid, stock_ids, adv, sigma, quantile, rebalance, cm, tradable=None,
           min_names=10, long_only=False, top_liquidity=1.0, pnl_parts=None, open_entry=False):
    """Replay the book and return the cost decomposition, in annualised return units."""
    dates = np.unique(di)
    order = np.argsort(di, kind="stable")
    pos = {d: i for i, d in enumerate(dates)}
    rows = []
    for d in dates:
        m = di == d
        if tradable is not None:
            m = m & tradable
        idxs = np.where(m)[0]
        if top_liquidity < 1.0 and idxs.size:
            # trade only the most liquid names of the day: the cheapest way to buy capacity, and
            # the universe a size-constrained book would actually be run on
            k = max(1, int(np.ceil(top_liquidity * idxs.size)))
            idxs = idxs[np.argsort(-adv[idxs])[:k]]
        rows.append((d, idxs))
    rows = [r for r in rows if r[1].size]

    prev_l, prev_s = {}, {}
    held_l, held_s = {}, {}
    kept_l, kept_s = set(), set()
    # the argument is called pnl_parts because `parts` is used below for the
    # participation bookkeeping, and the two names collided
    on_arr, id_arr = (pnl_parts if pnl_parts is not None else (None, None))
    gross_l, gross_s = [], []
    on_l, id_l, on_s, id_s, oe_l, oe_s = [], [], [], [], [], []
    fee_l, fee_s, imp_l, imp_s, bor = [], [], [], [], []
    warn_hits, warn_total, parts = 0, 0, []
    traded_all, traded_above = 0.0, 0.0
    for step, (d, idxs) in enumerate(rows):
        day = int(grid[d].astype("datetime64[D]").astype(np.int64))
        need = max(min_names, int(np.ceil(1.0 / quantile)) * 2)
        if (step % max(1, rebalance) == 0 or not held_l) and len(idxs) >= need:
            k = max(1, int(round(quantile * len(idxs))))
            o = np.argsort(-scores[idxs])
            held_l = {int(stock_ids[idxs[o[j]]]): (idxs[o[j]], 1.0 / k) for j in range(k)}
            held_s = ({} if long_only else
                      {int(stock_ids[idxs[o[-1 - j]]]): (idxs[o[-1 - j]], 1.0 / k)
                       for j in range(k)})
            t_buy_l = t_sell_l = t_buy_s = t_sell_s = 0.0
            f_l = f_s = 0.0
            for asset, (row, w) in held_l.items():
                pw = prev_l.get(asset, (row, 0.0))[1]
                if w > pw:
                    d_buy = w - pw
                    f_l += d_buy * cm.one_way(day, selling=False)
                    t_buy_l += d_buy
                elif w < pw:
                    d_sell = pw - w
                    f_l += d_sell * cm.one_way(day, selling=True)
                    t_sell_l += d_sell
            for asset in set(prev_l) - set(held_l):
                pw = prev_l[asset][1]
                f_l += pw * cm.one_way(day, selling=True)
                t_sell_l += pw
            if not long_only:
                for asset, (row, w) in held_s.items():
                    pw = prev_s.get(asset, (row, 0.0))[1]
                    if w > pw:
                        # opening or adding to a short is a sale, so stamp duty applies
                        d = w - pw
                        f_s += d * cm.one_way(day, selling=True)
                        t_buy_s += d
                    elif w < pw:
                        d = pw - w
                        f_s += d * cm.one_way(day, selling=False)
                        t_sell_s += d
                for asset in set(prev_s) - set(held_s):
                    pw = prev_s[asset][1]
                    f_s += pw * cm.one_way(day, selling=False)
                    t_buy_s += pw
            kept_l = set(prev_l) & set(held_l)
            kept_s = set(prev_s) & set(held_s)
            prev_l, prev_s = dict(held_l), dict(held_s)
            trades_l = (t_buy_l + t_sell_l)
            trades_s = (t_buy_s + t_sell_s) if not long_only else 0.0
        else:
            f_l = f_s = 0.0
            trades_l = trades_s = 0.0

        today = {int(stock_ids[i]): i for i in idxs}

        def impact(held, traded_weight):
            if traded_weight <= 0:
                return 0.0
            tot = 0.0
            for asset, (row, w) in held.items():
                if asset not in today:
                    continue
                i = today[asset]
                notional = w * cm.aum
                part = notional / max(adv[i], 1.0)
                parts.append(part)
                nonlocal traded_all, traded_above, warn_hits
                traded_all += w
                if part > cm.participation_warn:
                    warn_hits += 1
                    traded_above += w
                tot += w * cm.impact_coef * sigma[i] * np.sqrt(max(part, 0.0))
            return tot

        imp_l.append(impact(held_l, trades_l))
        imp_s.append(impact(held_s, trades_s) if not long_only else 0.0)
        fee_l.append(f_l)
        fee_s.append(f_s)
        warn_total += 1

        rl = np.mean([y[today[a]] for a in held_l if a in today]) if held_l else 0.0
        rs = (np.mean([y[today[a]] for a in held_s if a in today]) if held_s else 0.0)
        gross_l.append(float(rl))
        gross_s.append(float(rs))
        if pnl_parts is not None:
            hold = [a for a in held_l if a in today]
            on_l.append(float(np.mean([on_arr[today[a]] for a in hold])) if hold else 0.0)
            id_l.append(float(np.mean([id_arr[today[a]] for a in hold])) if hold else 0.0)
            if long_only or not held_s:
                on_s.append(0.0); id_s.append(0.0)
            else:
                hs = [a for a in held_s if a in today]
                on_s.append(float(np.mean([on_arr[today[a]] for a in hs])) if hs else 0.0)
                id_s.append(float(np.mean([id_arr[today[a]] for a in hs])) if hs else 0.0)
            if open_entry:
                # entering at the open of the target day means the overnight jump of a newly added
                # name is not earned, only its intraday return
                new = [a for a in hold if a not in kept_l]
                oe = float(np.mean([id_arr[today[a]] for a in hold] )) if hold else 0.0
                if new:
                    oe = float(np.mean([(id_arr[today[a]] if a in new else
                                         on_arr[today[a]] + id_arr[today[a]]) for a in hold]))
                oe_l.append(oe)
                hs = [a for a in held_s if a in today]
                new_s = [a for a in hs if a not in kept_s]
                oes = float(np.mean([id_arr[today[a]] for a in hs])) if hs else 0.0
                if new_s:
                    oes = float(np.mean([(id_arr[today[a]] if a in new_s else
                                          on_arr[today[a]] + id_arr[today[a]]) for a in hs]))
                oe_s.append(oes)
        bor.append(cm.borrow_annual / 252.0 if not long_only else 0.0)

    def ann(x):
        return float(np.nanmean(x) * 252 * 100)

    fee = ann(fee_l) + ann(fee_s)
    imp = ann(imp_l) + ann(imp_s)
    borrow = ann(bor)
    gross = ann(gross_l) - ann(gross_s)
    net = gross - fee - imp - borrow
    parts = np.asarray(parts) if parts else np.zeros(1)
    out_extra = {}
    if on_l:
        out_extra["gross_on_ann"] = ann(on_l) - ann(on_s)
        out_extra["gross_id_ann"] = ann(id_l) - ann(id_s)
    if oe_l:
        out_extra["gross_open_entry_ann"] = ann(oe_l) - ann(oe_s)
    return {"gross_ann": gross, "fee_ann": fee, "impact_ann": imp, "borrow_ann": borrow,
            "net_ann": net, "cost_ann": fee + imp + borrow, **out_extra,
            "participation_mean": float(parts.mean()), "participation_p95": float(np.percentile(parts, 95)),
            "participation_max": float(parts.max()),
            "traded_above_threshold_share": (traded_above / traded_all) if traded_all else 0.0,
            "days": len(gross_l)}


def breakeven_aum(gross, fee, borrow, impact_at_ref, aum_ref):
    """Book size at which impact consumes what is left of the gross return.

    Impact grows as the square root of size, so with ``impact = k sqrt(AUM)`` the zero crossing is
    ``((gross - fee - borrow) / k) ** 2``. Returns 0 when the strategy is already unprofitable
    before size is considered.
    """
    left = gross - fee - borrow
    if left <= 0 or impact_at_ref <= 0:
        return 0.0
    k = impact_at_ref / np.sqrt(aum_ref)
    return float((left / k) ** 2)
