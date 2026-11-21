"""Antarmuka Streamlit: monitoring, peringatan, histori, dan kontrol PID.

Tampilan mengikuti gambar referensi: header, bar status, dua kartu besar
(suhu & kelembapan), kotak peringatan + legenda, dan kartu histori data.
Status PID dan panel pengaturan dipindah ke bagian lipat (expander) di bawah.
"""
from __future__ import annotations

import hmac
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import alerts
from communication import MqttBridge
from config import get_settings
from database import NUMERIC_COLS, SupabaseClient
from pid_control import LIMIT_KEYS, ControlConfig, Limits, from_payload, validate

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
          "Agustus", "September", "Oktober", "November", "Desember"]
GREEN, YELLOW, RED, GREY = "#22c55e", "#f5b301", "#ef4444", "#64748b"
RANGES = {"1 jam": 1, "6 jam": 6, "12 jam": 12, "24 jam": 24, "Seluruh histori": None}
REASONS = {
    "OK": "",
    "SYSTEM_OFF": "Sistem OFF",
    "SENSOR_FAIL": "Fail-safe: sensor gagal",
    "COMM_LOSS": "Fail-safe: komunikasi hilang",
    "TEMP_CUTOFF": "Fail-safe: suhu di bawah batas aman",
}
TABLE_COLS = {"created_at": "Waktu", "temperature": "Suhu (°C)", "humidity": "RH (%)",
              "setpoint": "Setpoint (°C)", "error": "Error (°C)", "kp": "Kp", "ki": "Ki",
              "kd": "Kd", "duty": "Duty (%)", "actuator": "Aktuator", "mode": "Mode",
              "power": "Sistem ON"}

CSS = """
<style>
.stApp { background: #0b1220; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 960px; padding-top: 2rem; }

/* ---- header & bar status ---- */
.title { text-align:center; font-weight:800; font-size:clamp(1.15rem,2.6vw,1.9rem);
         letter-spacing:.5px; color:#f1f5f9; line-height:1.35; margin:.2rem 0 1.1rem; }
.sysbar { display:flex; justify-content:space-between; flex-wrap:wrap; gap:.4rem 1.2rem;
          font-size:.85rem; color:#cbd5e1; margin-bottom:.8rem; }
.dot { display:inline-block; width:.62rem; height:.62rem; border-radius:50%; margin-right:.35rem; }

/* ---- kartu ---- */
.card { background:#172236; border:1px solid #2a3750; border-radius:12px; padding:1rem 1.2rem; }
.metric { display:flex; justify-content:space-between; align-items:center; min-height:160px; margin-bottom:.6rem; }
.metric-title { color:#e2e8f0; font-weight:600; font-size:.95rem; }
.metric-value { font-size:clamp(2.4rem,5.5vw,3.5rem); font-weight:800; color:#fff; line-height:1.1;
                margin:.35rem 0 .5rem; text-shadow:0 0 18px rgba(255,255,255,.22); }
.unit { font-size:.55em; font-weight:700; }
.badge { display:inline-block; padding:.12rem .7rem; border-radius:6px; font-weight:700;
         font-size:.78rem; color:#0b1220; }
.note { color:#cbd5e1; font-size:.85rem; margin-top:.55rem; }
.bigicon { padding-left:.8rem; line-height:0; }

/* ---- peringatan ---- */
.sec-title { color:#e2e8f0; font-weight:700; font-size:.95rem; margin-bottom:.6rem; }
.alert-box { border-radius:8px; padding:.7rem 1rem; font-size:.9rem; color:#f8fafc; }
.legend { display:flex; flex-wrap:wrap; gap:.5rem 1.8rem; margin-top:.85rem; font-size:.82rem; color:#cbd5e1; }
.legend-note { color:#94a3b8; font-size:.74rem; margin-top:.55rem; line-height:1.5; }

/* ---- status PID ---- */
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.mini { background:#101a2c; border:1px solid #26334b; border-radius:10px; padding:.55rem .8rem; }
.mini-l { color:#94a3b8; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
.mini-v { color:#f1f5f9; font-size:1.05rem; font-weight:700; }
.foot { color:#64748b; font-size:.75rem; text-align:center; margin-top:1.5rem; }

/* ---- kartu histori (container bertanda .hist-marker) ---- */
.hist-marker { display:none; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) {
  background:#172236; border:1px solid #2a3750 !important; border-radius:12px; padding:.5rem .8rem .6rem; }
[data-testid="stExpander"] { background:#172236; border:1px solid #2a3750; border-radius:12px; }

/* ---- tombol terang seperti di gambar ---- */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stFormSubmitButton"] button {
  background:#e5e7eb; color:#0f172a; border:1px solid #cbd5e1; border-radius:6px;
  font-weight:600; min-height:0; padding:.3rem .9rem; }
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
  background:#ffffff; color:#0f172a; border-color:#ffffff; }
div[data-testid="stButton"] button p,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stFormSubmitButton"] button p { color:#0f172a; font-size:.82rem; font-weight:600; }
</style>
"""


# ------------------------------------------------------------------ util
def _html(s: str) -> str:
    return "".join(line.strip() for line in s.strip().splitlines())


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _fmt(v, fmt: str = "{:.1f}", dash: str = "--") -> str:
    return dash if v is None else fmt.format(v)


def _dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


# ------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Menghubungkan ke broker MQTT…")
def get_bridge() -> MqttBridge:
    bridge = MqttBridge(get_settings())
    bridge.wait_ready(4.0)
    return bridge


@st.cache_resource
def get_db():
    s = get_settings()
    if not s.db_configured:
        return None
    return SupabaseClient(s.supabase_url, s.supabase_key, s.timezone)


# ------------------------------------------------------------ ikon SVG
def _icon_thermo(color: str, value) -> str:
    """Termometer; tinggi cairan mengikuti nilai (skala 0–40 °C)."""
    frac = 0.0 if value is None else min(max(value / 40.0, 0.0), 1.0)
    level_y = 56 - frac * 34
    ticks = "".join(
        f'<line x1="48" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        for y, x2 in ((22, 56), (30, 53), (38, 56), (46, 53)))
    return (
        f'<svg width="64" height="92" viewBox="0 0 64 92" fill="none" xmlns="http://www.w3.org/2000/svg" '
        f'style="filter:drop-shadow(0 0 8px {_rgba(color, .6)})">'
        f'<path d="M22 16a10 10 0 0 1 20 0V57a18 18 0 1 1-20 0Z" stroke="{color}" stroke-width="3" '
        f'fill="{_rgba(color, .1)}"/>'
        f'<line x1="32" y1="62" x2="32" y2="{level_y:.1f}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="32" cy="72" r="10" fill="{color}"/>{ticks}</svg>')


def _icon_drop(color: str) -> str:
    return (
        f'<svg width="64" height="84" viewBox="0 0 64 84" fill="none" xmlns="http://www.w3.org/2000/svg" '
        f'style="filter:drop-shadow(0 0 8px {_rgba(color, .6)})">'
        f'<path d="M32 4C32 4 8 34 8 54a24 24 0 0 0 48 0C56 34 32 4 32 4Z" fill="{color}"/>'
        f'<path d="M20 56a12 12 0 0 0 9 11" stroke="#fff" stroke-opacity=".45" stroke-width="4" '
        f'stroke-linecap="round"/></svg>')


# ------------------------------------------------------------ komponen UI
def _metric_card(icon: str, title: str, value, unit: str, stt: alerts.Status, art: str) -> str:
    c = stt.color
    return _html(f"""
    <div class="card metric" style="border-color:{_rgba(c, .75)};box-shadow:0 0 22px {_rgba(c, .2)},inset 0 0 30px {_rgba(c, .05)};">
      <div>
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{_fmt(value)}<span class="unit"> {unit if value is not None else ''}</span></div>
        <span class="badge" style="background:{c}">● {stt.level}</span>
        <div class="note">{stt.note}</div>
      </div>
      <div class="bigicon">{art}</div>
    </div>""")


def _legend(l: Limits) -> str:
    items = [
        (GREEN, f"{l.temp_ideal_min:g} – {l.temp_ideal_max:g} °C", "NORMAL"),
        (YELLOW, f"&gt; {l.temp_ideal_max:g} – {l.temp_crit_max:g} °C", "WARNING"),
        (RED, f"&gt; {l.temp_crit_max:g} °C", "MASALAH / KRITIS"),
    ]
    row = '<div class="legend">' + "".join(
        f'<div>{_dot(c)}{rng} : <b>{name}</b></div>' for c, rng, name in items) + "</div>"
    note = (f'<div class="legend-note">Di luar rentang ideal dianggap WARNING (termasuk suhu &lt; {l.temp_ideal_min:g} °C); '
            f'suhu kritis bila &lt; {l.temp_crit_min:g} °C. Kelembapan: ideal {l.hum_ideal_min:g}–{l.hum_ideal_max:g} %RH, '
            f'kritis bila &lt; {l.hum_crit_min:g} atau &gt; {l.hum_crit_max:g} %RH.</div>')
    return row + note


def _mini(label: str, value: str) -> str:
    return f'<div class="mini"><div class="mini-l">{label}</div><div class="mini-v">{value}</div></div>'


@st.fragment(run_every=2)
def live_panel() -> None:
    s = get_settings()
    bridge = get_bridge()
    snap = bridge.snapshot()
    cfg_dict = snap["config"]
    cfg = from_payload(cfg_dict) if cfg_dict else ControlConfig()
    limits = cfg.limits
    tel = snap["telemetry"]            # None bila ESP32 offline → data lama tidak ditampilkan
    online = tel is not None
    sensor_ok = bool(tel.get("sensor_ok", False)) if tel else False
    temp = _num(tel.get("temp")) if tel else None
    hum = _num(tel.get("hum")) if tel else None

    # --- baris status
    now = datetime.now(ZoneInfo(s.timezone))
    esp_c, esp_t = (GREEN, "Online") if online else (RED, "Offline")
    if not online:
        dht_c, dht_t = GREY, "Tidak diketahui"
    elif sensor_ok:
        dht_c, dht_t = GREEN, "Terkoneksi (Online)"
    else:
        dht_c, dht_t = RED, "Error"
    broker_chip = "" if snap["broker_connected"] else \
        f'<span style="margin-right:1.1rem">{_dot(RED)}Broker: Terputus</span>'
    st.markdown(_html(f"""
      <div class="sysbar">
        <div>Waktu Sistem: <b>{now.day} {MONTHS[now.month - 1]} {now.year}</b> | {now:%H:%M} {now.tzname()}</div>
        <div>{broker_chip}<span style="margin-right:1.1rem">{_dot(esp_c)}ESP32: <span style="color:{esp_c}">{esp_t}</span></span>Status Sensor DHT: {_dot(dht_c)}<span style="color:{dht_c}">{dht_t}</span></div>
      </div>"""), unsafe_allow_html=True)
    if snap["last_error"]:
        st.warning(snap["last_error"])

    # --- kartu suhu & kelembapan
    t_st = alerts.temperature_status(temp, limits, online, sensor_ok)
    h_st = alerts.humidity_status(hum, limits, online, sensor_ok)
    c1, c2 = st.columns(2)
    c1.markdown(_metric_card("🌡️", "TEMPERATUR RUANG", temp, "°C", t_st, _icon_thermo(t_st.color, temp)),
                unsafe_allow_html=True)
    c2.markdown(_metric_card("💧", "KELEMBAPAN RUANG", hum, "%RH", h_st, _icon_drop(h_st.color)),
                unsafe_allow_html=True)

    # --- sistem peringatan
    pid_active = bool(tel and tel.get("mode") == "AUTO" and tel.get("power"))
    level, message = alerts.overall(t_st, h_st, temp, hum, limits, pid_active)
    ac = alerts.COLORS[level]
    st.markdown(_html(f"""
      <div class="card" style="margin-bottom:.8rem">
        <div class="sec-title">🚨 SISTEM PERINGATAN (ALERT)</div>
        <div class="alert-box" style="background:{_rgba(ac, .16)};border:1px solid {_rgba(ac, .7)}">
          {_dot(ac)}<b>{level}:</b> {message}
        </div>
        {_legend(limits)}
      </div>"""), unsafe_allow_html=True)


# --------------------------------------------------------------- histori
@st.cache_data(ttl=600, show_spinner=False, max_entries=16)
def load_history(hours, nonce: int) -> pd.DataFrame:  # nonce hanya untuk memaksa muat ulang
    return get_db().fetch_history(hours)


def _with_gaps(df: pd.DataFrame, gap_minutes: float) -> pd.DataFrame:
    """Sisipkan titik kosong di celah data agar garis putus (tidak menyambung semu)."""
    if len(df) < 2:
        return df
    d = df.sort_values("created_at").reset_index(drop=True)
    delta = d["created_at"].diff()
    gaps = delta > pd.Timedelta(minutes=gap_minutes)
    if not gaps.any():
        return d
    filler = d.loc[gaps].copy()
    filler["created_at"] = d["created_at"].shift(1)[gaps] + delta[gaps] / 2
    for c in NUMERIC_COLS:
        filler[c] = np.nan
    return pd.concat([d, filler]).sort_values("created_at").reset_index(drop=True)


def _chart(df: pd.DataFrame, gap_minutes: float, hours) -> go.Figure:
    d = _with_gaps(df, gap_minutes)
    many = len(d) > 1500
    mode = "lines" if many else "lines+markers"
    tickfmt = "%H:%M" if (hours is not None and hours <= 24) else "%d/%m %H:%M"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["created_at"], y=d["temperature"], name="Temperatur (°C)", mode=mode,
                             line=dict(color="#ef4444", width=2), marker=dict(size=4),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.28)", connectgaps=False))
    fig.add_trace(go.Scatter(x=d["created_at"], y=d["humidity"], name="Kelembapan (%)", mode=mode,
                             line=dict(color="#3b82f6", width=2), marker=dict(size=4),
                             fill="tozeroy", fillcolor="rgba(59,130,246,0.12)", connectgaps=False))
    # Setpoint hanya digambar bila memang ada datanya (firmware mode PID).
    if d["setpoint"].notna().any():
        fig.add_trace(go.Scatter(x=d["created_at"], y=d["setpoint"], name="Setpoint (°C)", mode="lines",
                                 line=dict(color="#f5b301", width=1.5, dash="dash"), connectgaps=False))
    fig.update_layout(
        template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="rgba(148,163,184,.12)", tickformat=tickfmt),
        yaxis=dict(gridcolor="rgba(148,163,184,.12)", rangemode="tozero"),
    )
    return fig


@st.fragment(run_every=30)
def history_panel() -> None:
    s = get_settings()
    st.session_state.setdefault("hist_nonce", 0)
    st.session_state.setdefault("show_table", False)

    with st.container(border=True):
        st.markdown('<span class="hist-marker"></span>', unsafe_allow_html=True)
        title_ph = st.empty()

        c1, c2, c3 = st.columns([3.4, 1.5, 1.1])
        label = c1.radio("Rentang waktu", list(RANGES), index=3, horizontal=True,
                         key="hist_range", label_visibility="collapsed")
        auto = c2.toggle("Auto-refresh grafik", value=True, key="hist_auto")
        if c3.button("↻ Muat ulang", key="hist_reload"):
            st.session_state.hist_nonce += 1
        hours = RANGES[label]
        title_ph.markdown(
            f'<div class="sec-title">📈 HISTORI DATA ({"REKAPAN " + label.upper() + " TERAKHIR" if hours else "SELURUH HISTORI"})</div>',
            unsafe_allow_html=True)

        if get_db() is None:
            st.warning("Database belum dikonfigurasi (SUPABASE_URL / SUPABASE_SERVICE_KEY).")
            return
        nonce = int(time.time() // 30) if auto else st.session_state.hist_nonce
        try:
            df = load_history(hours, nonce)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal membaca histori dari Supabase: {exc}")
            return
        if df.empty:
            st.info("Belum ada data tersimpan pada rentang ini. Pastikan ESP32 online dan tabel "
                    "`sensor_log` di Supabase sudah dibuat (lihat sql/schema.sql).")
            return

        st.plotly_chart(_chart(df, s.gap_minutes, hours), config={"displayModeBar": False})

        out = df.copy()
        out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        _, b1, b2 = st.columns([2.6, 1.5, 1.7])
        b1.download_button("Data Ekspor (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"histori_arsip_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv", key="dl_csv")
        table_label = "Sembunyikan Tabel Log" if st.session_state.show_table else "Tampilkan Tabel Log"
        if b2.button(table_label, key="btn_table"):
            st.session_state.show_table = not st.session_state.show_table
            st.rerun(scope="fragment")
        st.caption(f"{len(df):,} pembacaan · terakhir: {df['created_at'].iloc[-1]:%d-%m-%Y %H:%M:%S}")
        if st.session_state.show_table:
            view = df.sort_values("created_at", ascending=False).head(1000).copy()
            view["created_at"] = view["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(view.rename(columns=TABLE_COLS), hide_index=True)


# ---------------------------------------------------------- status PID
@st.fragment(run_every=2)
def pid_panel() -> None:
    snap = get_bridge().snapshot()
    cfg_dict = snap["config"]
    tel = snap["telemetry"]
    labels = ("Setpoint suhu", "Suhu aktual", "Error (aktual − setpoint)", "Kp / Ki / Kd",
              "Mode kendali", "Sistem", "Aktuator (Peltier)", "Konfigurasi di ESP32")
    if tel:
        duty = _num(tel.get("duty"))
        reason = REASONS.get(str(tel.get("reason", "OK")), str(tel.get("reason", "")))
        act = f"{tel.get('act', '--')} · {_fmt(duty, '{:.0f}')} %" + (f"<br><small>{reason}</small>" if reason else "")
        try:
            synced = cfg_dict is not None and int(tel.get("rev", -1)) == int(cfg_dict.get("rev", -2))
        except (TypeError, ValueError):
            synced = False
        values = [
            _fmt(_num(tel.get("sp")), "{:.1f} °C"),
            _fmt(_num(tel.get("temp")), "{:.1f} °C"),
            _fmt(_num(tel.get("err")), "{:+.1f} °C"),
            f"{_fmt(_num(tel.get('kp')), '{:g}')} / {_fmt(_num(tel.get('ki')), '{:g}')} / {_fmt(_num(tel.get('kd')), '{:g}')}",
            str(tel.get("mode", "--")),
            "ON" if tel.get("power") else "OFF",
            act,
            "✔ Sinkron" if synced else "⏳ Menunggu ESP32",
        ]
    else:
        values = ["--"] * len(labels)
    st.markdown(_html(f"""
      <div class="sec-title">🎛️ STATUS KENDALI PID</div>
      <div class="grid">{''.join(_mini(l, v) for l, v in zip(labels, values))}</div>"""),
        unsafe_allow_html=True)


# --------------------------------------------------------------- kontrol
def _active_cfg(bridge: MqttBridge) -> ControlConfig:
    d = bridge.snapshot()["config"]
    return from_payload(d) if d else ControlConfig()


def _apply_cfg_to_form(cfg: ControlConfig) -> None:
    ss = st.session_state
    ss.f_sp, ss.f_kp, ss.f_ki, ss.f_kd = float(cfg.setpoint), float(cfg.kp), float(cfg.ki), float(cfg.kd)
    ss.f_mode, ss.f_power = cfg.mode, bool(cfg.power)
    ss.f_man, ss.f_max = float(cfg.manual_duty), float(cfg.duty_max)
    for k in LIMIT_KEYS:
        ss[f"l_{k}"] = float(getattr(cfg.limits, k))


def _send(bridge: MqttBridge, cfg: ControlConfig) -> None:
    ok, msg = bridge.publish_config(cfg)
    if ok:
        st.success(msg + " ESP32 akan menerapkannya; cek kotak 'Konfigurasi di ESP32' = ✔ Sinkron.")
    else:
        st.error(msg)


def control_panel() -> None:
    s = get_settings()
    bridge = get_bridge()
    st.markdown('<div class="sec-title" style="margin-top:.4rem">⚙️ PENGATURAN & KONTROL SISTEM</div>',
                unsafe_allow_html=True)

    if s.dashboard_password and not st.session_state.get("auth_ok"):
        pw = st.text_input("Kata sandi panel kontrol", type="password", key="pw_input")
        if st.button("🔓 Buka panel kontrol"):
            if hmac.compare_digest(pw.encode(), s.dashboard_password.encode()):
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Kata sandi salah.")
        st.caption("Monitoring dapat dilihat tanpa kata sandi; pengaturan dikunci.")
        return

    if "f_sp" not in st.session_state:
        _apply_cfg_to_form(_active_cfg(bridge))

    q1, q2, q3 = st.columns(3)
    if q1.button("▶ Sistem ON"):
        cfg = _active_cfg(bridge)
        cfg.power = True
        st.session_state.f_power = True
        _send(bridge, cfg)
    if q2.button("⏹ Sistem OFF (darurat)"):
        cfg = _active_cfg(bridge)
        cfg.power = False
        st.session_state.f_power = False
        _send(bridge, cfg)
    q3.button("↻ Muat konfigurasi aktif", on_click=lambda: _apply_cfg_to_form(_active_cfg(bridge)))

    with st.form("pid_form"):
        a, b, c = st.columns(3)
        a.number_input("Setpoint temperatur (°C)", 5.0, 40.0, step=0.5, format="%.1f", key="f_sp")
        b.number_input("Kp (% duty per °C)", 0.0, 1000.0, step=0.5, format="%.3f", key="f_kp")
        c.number_input("Ki (% duty per °C·s)", 0.0, 1000.0, step=0.05, format="%.4f", key="f_ki")
        d, e, f = st.columns(3)
        d.number_input("Kd (% duty per °C/s)", 0.0, 1000.0, step=0.5, format="%.3f", key="f_kd")
        e.radio("Mode kendali", ["AUTO", "MANUAL"], horizontal=True, key="f_mode")
        f.toggle("Sistem ON", key="f_power")
        g, h = st.columns(2)
        g.slider("Duty manual aktuator (%) — dipakai saat MANUAL", 0.0, 100.0, step=1.0, key="f_man")
        h.slider("Batas duty maksimum PWM (%) — proteksi Peltier", 0.0, 100.0, step=1.0, key="f_max")
        with st.expander("Batas peringatan (alert)"):
            t, r = st.columns(2)
            t.number_input("Suhu ideal min (°C)", step=0.5, key="l_temp_ideal_min")
            t.number_input("Suhu ideal maks (°C)", step=0.5, key="l_temp_ideal_max")
            t.number_input("Suhu kritis min (°C)", step=0.5, key="l_temp_crit_min")
            t.number_input("Suhu kritis maks (°C)", step=0.5, key="l_temp_crit_max")
            r.number_input("RH ideal min (%)", step=1.0, key="l_hum_ideal_min")
            r.number_input("RH ideal maks (%)", step=1.0, key="l_hum_ideal_max")
            r.number_input("RH kritis min (%)", step=1.0, key="l_hum_crit_min")
            r.number_input("RH kritis maks (%)", step=1.0, key="l_hum_crit_max")
        submitted = st.form_submit_button("💾 Kirim & Simpan Pengaturan")

    if submitted:
        ss = st.session_state
        cfg = ControlConfig(
            setpoint=ss.f_sp, kp=ss.f_kp, ki=ss.f_ki, kd=ss.f_kd, mode=ss.f_mode,
            power=ss.f_power, manual_duty=ss.f_man, duty_max=ss.f_max,
            limits=Limits(**{k: ss[f"l_{k}"] for k in LIMIT_KEYS}),
        )
        errs = validate(cfg)
        if errs:
            for m in errs:
                st.error(m)
        else:
            _send(bridge, cfg)
    st.caption("Pengaturan disimpan di broker (retained) sehingga otomatis dikirim ulang ke ESP32 saat menyala "
               "kembali. Pin, frekuensi PWM, dan jenis aktuator diatur di firmware (esp32/config.h).")


# ---------------------------------------------------------------- entry
def render() -> None:
    s = get_settings()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="title">SISTEM KENDALI PID IKLIM MIKRO RUANG ARSIP BERBASIS IoT<br>'
                'DENGAN ANTARMUKA PEMANTAUAN REAL-TIME</div>', unsafe_allow_html=True)
    if not s.mqtt_configured:
        st.error("Kredensial MQTT belum diisi. Salin `.streamlit/secrets.toml.example` menjadi "
                 "`.streamlit/secrets.toml` lalu isi MQTT_HOST, MQTT_USERNAME, dan MQTT_PASSWORD.")
        st.stop()
    live_panel()
    history_panel()
    with st.expander("🎛️ Status kendali PID & pengaturan sistem", expanded=False):
        pid_panel()
        control_panel()
    st.markdown('<div class="foot">Dashboard monitoring & pengaturan — algoritma PID berjalan di ESP32.</div>',
                unsafe_allow_html=True)
