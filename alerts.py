"""Logika sistem peringatan: NORMAL / WARNING / CRITICAL."""
from __future__ import annotations

from dataclasses import dataclass

from pid_control import Limits

COLORS = {
    "NORMAL": "#22c55e",
    "WARNING": "#f5b301",
    "CRITICAL": "#ef4444",
    "OFFLINE": "#64748b",
    "ERROR": "#ef4444",
}
_RANK = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass(frozen=True)
class Status:
    level: str
    note: str
    direction: str | None = None  # "tinggi" | "rendah" | None

    @property
    def color(self) -> str:
        return COLORS[self.level]


def _classify(v: float, imin: float, imax: float, cmin: float, cmax: float):
    if v < cmin or v > cmax:
        return "CRITICAL", ("tinggi" if v > cmax else "rendah")
    if v < imin or v > imax:
        return "WARNING", ("tinggi" if v > imax else "rendah")
    return "NORMAL", None


def temperature_status(value, limits: Limits, online: bool, sensor_ok: bool) -> Status:
    if not online:
        return Status("OFFLINE", "ESP32 offline — tidak ada data terkini.")
    if value is None or not sensor_ok:
        return Status("ERROR", "Sensor DHT22 tidak terbaca.")
    lvl, d = _classify(value, limits.temp_ideal_min, limits.temp_ideal_max,
                       limits.temp_crit_min, limits.temp_crit_max)
    if lvl == "NORMAL":
        note = "Suhu berada dalam batas ideal."
    elif lvl == "WARNING":
        note = "Suhu melebihi batas ideal." if d == "tinggi" else "Suhu di bawah batas ideal."
    else:
        note = ("Suhu KRITIS: melebihi batas kritis." if d == "tinggi"
                else "Suhu KRITIS: di bawah batas kritis.")
    return Status(lvl, note, d)


def humidity_status(value, limits: Limits, online: bool, sensor_ok: bool) -> Status:
    if not online:
        return Status("OFFLINE", "ESP32 offline — tidak ada data terkini.")
    if value is None or not sensor_ok:
        return Status("ERROR", "Sensor DHT22 tidak terbaca.")
    lvl, d = _classify(value, limits.hum_ideal_min, limits.hum_ideal_max,
                       limits.hum_crit_min, limits.hum_crit_max)
    if lvl == "NORMAL":
        note = "Kelembapan optimal."
    elif lvl == "WARNING":
        note = ("Kelembapan melebihi batas ideal." if d == "tinggi"
                else "Kelembapan di bawah batas ideal.")
    else:
        note = ("Kelembapan KRITIS: melebihi batas kritis." if d == "tinggi"
                else "Kelembapan KRITIS: di bawah batas kritis.")
    return Status(lvl, note, d)


def overall(temp_st: Status, hum_st: Status, temp, hum, limits: Limits,
            pid_active: bool) -> tuple[str, str]:
    """Kembalikan (level, pesan) untuk kotak Sistem Peringatan."""
    levels = {temp_st.level, hum_st.level}
    if "OFFLINE" in levels:
        return "OFFLINE", ("ESP32 tidak terhubung. Data terkini tidak tersedia, sehingga kondisi "
                           "ruang arsip tidak dapat dievaluasi.")
    if "ERROR" in levels:
        return "CRITICAL", ("Sensor DHT22 tidak terbaca. Kondisi ruangan tidak dapat dipantau; "
                            "aktuator dimatikan oleh proteksi firmware.")
    level = max((temp_st.level, hum_st.level), key=lambda x: _RANK[x])
    if level == "NORMAL":
        return "NORMAL", "Kondisi suhu dan kelembapan ruang arsip berada dalam batas yang ditentukan."

    parts: list[str] = []
    if temp_st.level != "NORMAL":
        arah = "melebihi" if temp_st.direction == "tinggi" else "berada di bawah"
        parts.append(f"Temperatur ruangan {temp:.1f} °C {arah} batas ideal "
                     f"({limits.temp_ideal_min:g} – {limits.temp_ideal_max:g} °C).")
        if temp_st.direction == "tinggi" and pid_active:
            parts.append("Sistem PID sedang bekerja untuk menstabilkan suhu kembali ke angka normal.")
    if hum_st.level != "NORMAL":
        arah = "melebihi" if hum_st.direction == "tinggi" else "berada di bawah"
        parts.append(f"Kelembapan ruangan {hum:.1f} %RH {arah} batas ideal "
                     f"({limits.hum_ideal_min:g} – {limits.hum_ideal_max:g} %RH). "
                     "Kelembapan hanya dipantau (tidak dikendalikan PID).")
    if level == "CRITICAL":
        parts.append("Diperlukan tindakan segera.")
    return level, " ".join(parts)
