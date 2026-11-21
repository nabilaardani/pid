"""Model konfigurasi kendali + validasi.

PENTING: algoritma PID TIDAK dihitung di sini. PID berjalan di firmware ESP32.
Modul ini hanya mendefinisikan paket konfigurasi yang dikirim dashboard ke
ESP32 lewat MQTT, dan memvalidasinya sebelum dikirim.

Konvensi: error = suhu aktual - setpoint (positif = ruangan lebih panas dari
setpoint, sehingga Peltier perlu mendinginkan).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any

SETPOINT_RANGE = (5.0, 40.0)


@dataclass
class Limits:
    """Batas peringatan. Nilai awal dapat diubah dari dashboard."""
    temp_ideal_min: float = 16.0
    temp_ideal_max: float = 24.0
    temp_crit_min: float = 10.0
    temp_crit_max: float = 30.0
    hum_ideal_min: float = 40.0
    hum_ideal_max: float = 60.0
    hum_crit_min: float = 30.0
    hum_crit_max: float = 70.0


LIMIT_KEYS = [f.name for f in fields(Limits)]


@dataclass
class ControlConfig:
    setpoint: float = 22.0
    kp: float = 20.0        # satuan: % duty per °C
    ki: float = 0.5         # % duty per (°C·detik)
    kd: float = 0.0         # % duty per (°C/detik)
    mode: str = "AUTO"      # AUTO | MANUAL
    power: bool = True      # ON/OFF sistem
    manual_duty: float = 0.0
    duty_max: float = 50.0  # batas atas duty PWM (proteksi Peltier)
    limits: Limits = field(default_factory=Limits)
    rev: int = 0


def validate(cfg: ControlConfig) -> list[str]:
    errs: list[str] = []
    lo, hi = SETPOINT_RANGE
    if not lo <= cfg.setpoint <= hi:
        errs.append(f"Setpoint harus antara {lo:g} dan {hi:g} °C.")
    for name, val in (("Kp", cfg.kp), ("Ki", cfg.ki), ("Kd", cfg.kd)):
        if not 0 <= val <= 1000:
            errs.append(f"{name} harus antara 0 dan 1000.")
    if cfg.mode not in ("AUTO", "MANUAL"):
        errs.append("Mode harus AUTO atau MANUAL.")
    if not 0 <= cfg.duty_max <= 100:
        errs.append("Duty maksimum harus 0–100 %.")
    if not 0 <= cfg.manual_duty <= 100:
        errs.append("Duty manual harus 0–100 %.")
    elif cfg.mode == "MANUAL" and cfg.manual_duty > cfg.duty_max:
        errs.append("Duty manual melebihi batas duty maksimum.")
    L = cfg.limits
    if not L.temp_crit_min < L.temp_ideal_min < L.temp_ideal_max < L.temp_crit_max:
        errs.append("Batas suhu harus berurutan: kritis min < ideal min < ideal maks < kritis maks.")
    if not L.hum_crit_min < L.hum_ideal_min < L.hum_ideal_max < L.hum_crit_max:
        errs.append("Batas kelembapan harus berurutan: kritis min < ideal min < ideal maks < kritis maks.")
    return errs


def to_payload(cfg: ControlConfig, rev: int | None = None) -> dict[str, Any]:
    """Format JSON yang dibaca firmware ESP32 (kunci singkat agar hemat memori)."""
    return {
        "sp": round(cfg.setpoint, 2),
        "kp": cfg.kp,
        "ki": cfg.ki,
        "kd": cfg.kd,
        "mode": cfg.mode,
        "power": bool(cfg.power),
        "man": cfg.manual_duty,
        "max": cfg.duty_max,
        "rev": int(rev if rev is not None else time.time()),
        "limits": {k: getattr(cfg.limits, k) for k in LIMIT_KEYS},
    }


def _f(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def from_payload(d: dict[str, Any]) -> ControlConfig:
    base = ControlConfig()
    lim_src = d.get("limits") or {}
    limits = Limits(**{k: _f(lim_src, k, getattr(base.limits, k)) for k in LIMIT_KEYS})
    mode = str(d.get("mode", base.mode)).upper()
    try:
        rev = int(d.get("rev", 0))
    except (TypeError, ValueError):
        rev = 0
    return ControlConfig(
        setpoint=_f(d, "sp", base.setpoint),
        kp=_f(d, "kp", base.kp),
        ki=_f(d, "ki", base.ki),
        kd=_f(d, "kd", base.kd),
        mode=mode if mode in ("AUTO", "MANUAL") else "AUTO",
        power=bool(d.get("power", base.power)),
        manual_duty=_f(d, "man", base.manual_duty),
        duty_max=_f(d, "max", base.duty_max),
        limits=limits,
        rev=rev,
    )
