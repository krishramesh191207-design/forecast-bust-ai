"""A deliberately simple stochastic toy atmosphere for the SANDBOX mode.

WHAT THIS IS
------------
A low-order, physically motivated caricature of the South Asian monsoon
region: a seasonally varying monsoon trough, propagating synoptic systems
(monsoon lows/depressions, tropical cyclones, western disturbances), a
moisture field, and precipitation generated from moisture-flux convergence,
orographic uplift and stochastic convection.

WHAT THIS IS NOT
----------------
It is NOT real weather and must never be presented as such. It exists so that
the *entire* forecast-verification, error-learning and explanation pipeline can
be executed end-to-end when external NWP/observation archives are not
reachable from the development environment.

WHY IT IS STILL USEFUL
----------------------
The "truth" run and the "forecast" runs share the same dynamics but differ in:
  * initial condition errors on system position/intensity,
  * independent stochastic convective triggering (irreducible error),
  * deliberate model biases (orographic precipitation deficit, dry bias over
    land, under-deepening of cyclones, warm bias over dry land).

Forecast error therefore *emerges* from the simulation rather than being
prescribed, so a machine-learning model that predicts forecast busts from
forecast-side features is solving a genuine (if idealised) inference problem.
No error or bust label anywhere in this project is written by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as Date

import numpy as np
from scipy.ndimage import gaussian_filter

from ..grid import gradient, mesh

RHO = 1.2
OMEGA = 7.292e-5


@dataclass
class System:
    kind: str           # monsoon_low | tropical_cyclone | western_disturbance
    lat: float
    lon: float
    depth: float        # hPa below environment at the centre
    radius: float       # degrees
    u_steer: float      # deg lon / day
    v_steer: float      # deg lat / day
    age: int = 0
    max_age: int = 8
    peak: float = 10.0


@dataclass
class State:
    systems: list[System] = field(default_factory=list)
    mjo_phase: float = 0.0        # radians, drives active/break cycle
    mjo_amp: float = 1.0
    heat: float = 0.0             # heat-wave amplitude over NW India (K)
    jet: float = 0.0              # subtropical jet anomaly (winter)

    def copy(self) -> "State":
        return State(
            systems=[System(**vars(s)) for s in self.systems],
            mjo_phase=self.mjo_phase,
            mjo_amp=self.mjo_amp,
            heat=self.heat,
            jet=self.jet,
        )


def _doy(d: Date) -> int:
    return d.timetuple().tm_yday


def _monsoon_envelope(doy: int) -> float:
    """0 in winter, 1 at the height of the summer monsoon (mid-July)."""
    return float(np.clip(math.sin(math.pi * (doy - 150) / 150.0), 0.0, 1.0)) if 150 <= doy <= 300 else 0.0


def _winter_envelope(doy: int) -> float:
    x = math.cos(2 * math.pi * (doy - 15) / 365.25)
    return float(np.clip((x - 0.1) / 0.9, 0.0, 1.0))


def _premonsoon_heat_envelope(doy: int) -> float:
    return float(np.clip(math.sin(math.pi * (doy - 75) / 110.0), 0.0, 1.0)) if 75 <= doy <= 185 else 0.0


class ToyAtmosphere:
    """Grid-aware field generator plus a daily state-evolution operator."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray, terrain: np.ndarray, land: np.ndarray):
        self.lats, self.lons = lats, lons
        self.terrain = terrain.astype("float32")
        self.land = land
        self.la, self.lo = mesh(lats, lons)
        self.f = 2 * OMEGA * np.sin(np.deg2rad(self.la))
        self.f_eff = np.sign(self.f) * np.maximum(np.abs(self.f), 2.0e-5)
        self.dhdx, self.dhdy = gradient(self.terrain, lats, lons)
        # distance from the coast used as a crude moisture-source proxy
        self.sea = (~land).astype("float32")
        self.sea_influence = gaussian_filter(self.sea, sigma=2.5, mode="nearest")

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------
    def spin_up(self, d: Date, rng: np.random.Generator) -> State:
        st = State(mjo_phase=rng.uniform(0, 2 * math.pi), mjo_amp=rng.uniform(0.6, 1.4))
        for _ in range(20):
            st = self.advance(st, d, rng)
        return st

    def advance(
        self,
        state: State,
        d: Date,
        rng: np.random.Generator,
        model_error: bool = False,
    ) -> State:
        """Advance the state by one day.

        `model_error=True` is used by forecast members: steering and intensity
        tendencies get an extra error term, representing imperfect dynamics.
        """
        st = state.copy()
        doy = _doy(d)
        mon = _monsoon_envelope(doy)
        win = _winter_envelope(doy)
        heat_env = _premonsoon_heat_envelope(doy)

        # --- large scale indices ---
        st.mjo_phase = (st.mjo_phase + 2 * math.pi / rng.uniform(35, 55)) % (2 * math.pi)
        st.mjo_amp = float(np.clip(st.mjo_amp + rng.normal(0, 0.16 if not model_error else 0.30), 0.2, 2.0))
        st.jet = float(np.clip(0.8 * st.jet + rng.normal(0, 0.5) + 1.2 * win, -2.5, 3.5))
        drive = heat_env * (1.0 - 0.6 * mon)
        st.heat = float(np.clip(0.85 * st.heat + rng.normal(0.35 * drive, 0.55 + 0.55 * model_error), -2.0, 8.0))

        # --- propagate existing systems ---
        alive: list[System] = []
        for s in st.systems:
            noise = 2.1 if model_error else 1.0
            s.lon += s.u_steer + rng.normal(0, 0.45 * noise)
            s.lat += s.v_steer + rng.normal(0, 0.30 * noise)
            s.age += 1
            # intensity life cycle: grow to peak then decay, faster over land
            over_land = self._is_land_point(s.lat, s.lon)
            growth = 1.0 - 2.0 * abs(s.age / max(s.max_age, 1) - 0.4)
            s.depth += 0.55 * growth * s.peak / max(s.max_age, 1) + rng.normal(0, 0.35 * noise)
            if over_land and s.kind in ("monsoon_low", "tropical_cyclone"):
                s.depth -= 0.9 if s.kind == "monsoon_low" else 2.4
            if model_error and s.kind == "tropical_cyclone":
                s.depth -= 0.45          # models under-deepen cyclones
            s.depth = max(s.depth, 0.0)
            s.radius = max(2.0, s.radius + rng.normal(0, 0.08))
            inside = (
                self.lats[0] - 4 <= s.lat <= self.lats[-1] + 4
                and self.lons[0] - 4 <= s.lon <= self.lons[-1] + 6
            )
            if s.depth > 0.6 and s.age <= s.max_age + 4 and inside:
                alive.append(s)
        st.systems = alive

        # --- genesis ---
        active = 0.5 + 0.5 * math.sin(st.mjo_phase) * st.mjo_amp
        if rng.random() < 0.16 * mon * max(active, 0.15):
            st.systems.append(
                System(
                    kind="monsoon_low",
                    lat=rng.uniform(17.0, 22.0),
                    lon=rng.uniform(84.0, 92.0),
                    depth=rng.uniform(1.5, 3.0),
                    radius=rng.uniform(4.0, 6.5),
                    u_steer=rng.uniform(-3.4, -1.6),
                    v_steer=rng.uniform(-0.1, 0.7),
                    max_age=int(rng.integers(5, 10)),
                    peak=rng.uniform(4.0, 12.0),
                )
            )
        tc_season = 1.0 if (100 <= doy <= 160 or 275 <= doy <= 345) else 0.15
        if rng.random() < 0.022 * tc_season:
            bay = rng.random() < 0.65
            st.systems.append(
                System(
                    kind="tropical_cyclone",
                    lat=rng.uniform(8.0, 14.0),
                    lon=rng.uniform(85.0, 92.0) if bay else rng.uniform(63.0, 71.0),
                    depth=rng.uniform(2.0, 4.0),
                    radius=rng.uniform(2.5, 4.5),
                    u_steer=rng.uniform(-2.2, 0.4) if bay else rng.uniform(-0.8, 1.4),
                    v_steer=rng.uniform(0.3, 1.5),
                    max_age=int(rng.integers(6, 12)),
                    peak=rng.uniform(12.0, 34.0),
                )
            )
        if rng.random() < 0.20 * win:
            st.systems.append(
                System(
                    kind="western_disturbance",
                    lat=rng.uniform(28.0, 36.0),
                    lon=rng.uniform(58.0, 64.0),
                    depth=rng.uniform(1.0, 2.5),
                    radius=rng.uniform(5.0, 8.0),
                    u_steer=rng.uniform(3.5, 6.5),
                    v_steer=rng.uniform(-0.4, 0.4),
                    max_age=int(rng.integers(5, 9)),
                    peak=rng.uniform(3.0, 9.0),
                )
            )
        return st

    def _is_land_point(self, lat: float, lon: float) -> bool:
        i = int(np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 1))
        j = int(np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 1))
        return bool(self.land[i, j])

    # ------------------------------------------------------------------
    # Field diagnosis
    # ------------------------------------------------------------------
    def _noise(self, rng: np.random.Generator, sigma: float) -> np.ndarray:
        """Unit-variance, spatially correlated random field."""
        z = gaussian_filter(rng.normal(0.0, 1.0, size=self.la.shape), sigma, mode="wrap")
        return z / max(float(z.std()), 1e-6)

    def fields(
        self,
        state: State,
        d: Date,
        rng: np.random.Generator,
        model_bias: bool = False,
    ) -> dict[str, np.ndarray]:
        doy = _doy(d)
        mon = _monsoon_envelope(doy)
        win = _winter_envelope(doy)
        active = 0.5 + 0.5 * math.sin(state.mjo_phase) * state.mjo_amp   # ~0..1.5

        # ---- mean sea level pressure -----------------------------------
        mslp = 1011.0 + 3.0 * win - 2.0 * mon + 0.02 * (self.la - 20.0)
        trough = np.exp(-((self.la - (21.0 + 2.0 * np.sin(np.deg2rad(self.lo - 70.0)))) ** 2) / (2 * 3.2**2))
        mslp -= (4.5 * mon * (0.55 + 0.75 * active)) * trough
        heat_low = np.exp(-(((self.la - 28.0) ** 2) / (2 * 3.0**2) + ((self.lo - 72.0) ** 2) / (2 * 4.0**2)))
        mslp -= 0.45 * state.heat * heat_low
        for s in state.systems:
            d2 = (self.la - s.lat) ** 2 + (self.lo - s.lon) ** 2
            mslp -= s.depth * np.exp(-d2 / (2 * s.radius**2))

        # ---- winds: geostrophic from MSLP + monsoon low-level jet -------
        dpdx, dpdy = gradient(mslp * 100.0, self.lats, self.lons)
        u = -dpdy / (RHO * self.f_eff)
        v = dpdx / (RHO * self.f_eff)
        jet = 12.0 * mon * (0.5 + 0.6 * active) * np.exp(
            -(((self.la - 12.0) ** 2) / (2 * 6.0**2) + ((self.lo - 66.0) ** 2) / (2 * 9.0**2))
        )
        u = u + jet
        westerly = (2.5 + 2.2 * state.jet) * np.exp(-((self.la - 31.0) ** 2) / (2 * 5.0**2)) * win
        u = u + westerly
        u = gaussian_filter(np.clip(u, -45, 45), 0.8, mode="nearest")
        v = gaussian_filter(np.clip(v, -45, 45), 0.8, mode="nearest")
        u10, v10 = 0.72 * u, 0.72 * v
        u850, v850 = u, v
        u200 = 0.4 * u + (18.0 + 9.0 * state.jet) * win * np.exp(-((self.la - 29.0) ** 2) / (2 * 5.5**2))
        u200 = u200 - 14.0 * mon * np.exp(-((self.la - 10.0) ** 2) / (2 * 7.0**2))   # tropical easterly jet

        # ---- moisture ---------------------------------------------------
        tcwv = (
            16.0
            + 34.0 * mon * (0.45 + 0.65 * active) * self.sea_influence
            + 12.0 * self.sea_influence
            + 8.0 * np.exp(-((self.la - 12.0) ** 2) / (2 * 9.0**2))
        )
        tcwv *= np.exp(-np.maximum(self.terrain, 0) / 3200.0)
        for s in state.systems:
            d2 = (self.la - s.lat) ** 2 + (self.lo - s.lon) ** 2
            amp = 14.0 if s.kind != "western_disturbance" else 5.0
            tcwv += amp * (s.depth / 8.0) * np.exp(-d2 / (2 * (s.radius * 1.3) ** 2))
        tcwv -= 6.0 * (1 - mon) * self.land * np.exp(-((self.la - 26.0) ** 2) / (2 * 8.0**2))
        if model_bias:
            tcwv -= 2.5 * self.land          # NWP dry bias over land
        tcwv = np.clip(tcwv, 3.0, 75.0)

        # ---- instability -------------------------------------------------
        cape = np.clip(
            58.0 * (tcwv - 22.0) * (0.35 + 0.65 * self.land) + 240.0 * mon * active - 260.0 * (1 - mon),
            0.0,
            4200.0,
        )
        if model_bias:
            cape *= 0.92

        # ---- precipitation -------------------------------------------------
        # Moisture-flux convergence term. conv has units kg m-2 s-1 == mm s-1,
        # multiplied by 86400 s/day and a precipitation efficiency.
        qu, qv = tcwv * u850, tcwv * v850
        dqudx, _ = gradient(qu, self.lats, self.lons)
        _, dqvdy = gradient(qv, self.lats, self.lons)
        conv = gaussian_filter(-(dqudx + dqvdy), 0.9, mode="nearest")
        dynamic = np.clip(conv, 0, None) * 86400.0 * 0.32

        # Orographic term: upslope flow lifts a layer of depth H_SCALE.
        upslope = np.clip(u850 * self.dhdx + v850 * self.dhdy, 0, None)
        orog_factor = 0.72 if model_bias else 1.0    # models under-do orographic rain
        orographic = orog_factor * upslope * tcwv / 3000.0 * 86400.0 * 0.30

        trigger = np.clip((cape - 550.0) / 1100.0, 0, None)
        # spatially correlated stochastic convection: the irreducible part of
        # the precipitation forecast problem.
        z = gaussian_filter(rng.normal(0.0, 1.0, size=self.la.shape), 1.7, mode="wrap")
        z = z / max(z.std(), 1e-6)
        noise = np.exp(0.62 * z - 0.19)
        convective = 7.5 * trigger * (0.4 + 0.6 * self.sea_influence + 0.35 * self.land) * noise

        precip = (dynamic + orographic) * (0.55 + 0.45 * noise) + convective
        precip = np.clip(gaussian_filter(precip, 0.6, mode="nearest"), 0.0, 400.0)

        # ---- temperature (after precipitation: clouds and rain cool) -----
        seasonal = 302.0 - 0.45 * np.abs(self.la - 12.0) + 6.5 * math.cos(2 * math.pi * (doy - 195) / 365.25) * (
            0.3 + 0.03 * np.maximum(self.la - 10.0, 0)
        )
        t2m = seasonal - 6.5 * np.maximum(self.terrain, 0) / 1000.0
        t2m += 1.2 * self.land * (1 - mon) - 1.0 * (1 - self.land)
        t2m += state.heat * heat_low * self.land
        t2m -= (4.5 * self.land + 1.2) * np.tanh(precip / 14.0)     # rain/cloud cooling
        t2m -= 1.8 * mon * self.sea_influence * active * self.land
        if model_bias:
            t2m += 0.8 * self.land * (1 - mon)   # warm bias over dry land

        # ---- upper air ---------------------------------------------------
        z500 = 5870.0 + 12.0 * (25.0 - self.la) + 55.0 * math.cos(2 * math.pi * (doy - 15) / 365.25) * np.clip(
            (self.la - 10.0) / 25.0, 0, 1
        )
        for s in state.systems:
            d2 = (self.la - s.lat) ** 2 + (self.lo - s.lon) ** 2
            z500 -= 8.5 * s.depth * np.exp(-d2 / (2 * (s.radius * 1.35) ** 2))
        z500 -= 18.0 * mon * trough
        t850 = t2m - 9.0 + 0.5 * np.maximum(self.terrain, 0) / 1000.0
        rh700 = np.clip(18.0 + 1.45 * tcwv + 12.0 * mon * active, 5.0, 100.0)
        shear = np.abs(u200 - u850) + np.abs(0.3 * v850)

        # ---- unresolved (sub-grid / stochastic-physics) variability -------
        # Spatially correlated variability that neither the truth run nor any
        # forecast member can reproduce in the other. Its amplitude depends on
        # the regime (moist, unstable, windy situations are less predictable),
        # which is exactly the kind of structure the bust model must learn.
        unpredictability = (
            0.35
            + 0.85 * np.clip(cape / 2200.0, 0, 1.6)
            + 0.9 * np.clip(tcwv / 55.0, 0, 1.4) * mon
            + 0.5 * np.clip(np.hypot(u850, v850) / 18.0, 0, 1.5)
        )
        t2m = t2m + 1.35 * unpredictability * (0.45 + 0.75 * self.land) * self._noise(rng, 2.2)
        u10 = u10 + 1.5 * unpredictability * self._noise(rng, 2.0)
        v10 = v10 + 1.5 * unpredictability * self._noise(rng, 2.0)
        u850 = u850 + 1.9 * unpredictability * self._noise(rng, 2.0)
        v850 = v850 + 1.9 * unpredictability * self._noise(rng, 2.0)
        mslp = mslp + 0.45 * unpredictability * self._noise(rng, 3.0)
        z500 = z500 + 9.0 * unpredictability * self._noise(rng, 3.0)
        tcwv = np.clip(tcwv + 2.6 * unpredictability * self._noise(rng, 2.4), 3.0, 80.0)

        return {
            "mslp": mslp.astype("float32"),
            "u10": u10.astype("float32"),
            "v10": v10.astype("float32"),
            "t2m": t2m.astype("float32"),
            "tcwv": tcwv.astype("float32"),
            "cape": cape.astype("float32"),
            "precip": precip.astype("float32"),
            "z500": z500.astype("float32"),
            "u850": u850.astype("float32"),
            "v850": v850.astype("float32"),
            "t850": t850.astype("float32"),
            "rh700": rh700.astype("float32"),
            "shear": shear.astype("float32"),
            "u200": u200.astype("float32"),
        }

    # ------------------------------------------------------------------
    def perturb_initial(self, state: State, rng: np.random.Generator, scale: float = 1.0) -> State:
        """Initial-condition perturbation for an ensemble member."""
        st = state.copy()
        for s in st.systems:
            s.lat += rng.normal(0, 0.85 * scale)
            s.lon += rng.normal(0, 1.10 * scale)
            s.depth = max(0.0, s.depth * rng.normal(1.0, 0.20 * scale))
            s.u_steer += rng.normal(0, 0.62 * scale)
            s.v_steer += rng.normal(0, 0.46 * scale)
        st.mjo_amp *= rng.normal(1.0, 0.16 * scale)
        st.heat += rng.normal(0, 0.9 * scale)
        st.jet += rng.normal(0, 0.6 * scale)
        return st
