
"""Antarmuka Streamlit: monitoring, peringatan, dan histori data.

Dashboard menampilkan:
- Header dan status koneksi
- Kartu temperatur dan kelembapan
- Sistem peringatan dan legenda
- Grafik histori sensor
- Ekspor CSV dan tabel log sederhana

Kontrol PID tetap berjalan di ESP32.
Panel status PID dan pengaturan tidak ditampilkan pada dashboard.
"""
from __future__ import annotations

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
from pid_control import ControlConfig, Limits, from_payload

MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

GREEN, YELLOW, RED, GREY = "#22c55e", "#f5b301", "#ef4444", "#64748b"

RANGES = {
    "1 jam": 1,
    "6 jam": 6,
    "12 jam": 12,
    "24 jam": 24,
    "Seluruh histori": None,
}

TABLE_COLS = {
    "created_at": "Waktu",
    "temperature": "Suhu (°C)",
    "humidity": "Kelembapan (%RH)",
}

CSS = """
<style>
.stApp {
    background: #0b1220;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 960px;
    padding-top: 2rem;
}

/* Header */
.title {
    text-align: center;
    font-weight: 800;
    font-size: clamp(1.15rem, 2.6vw, 1.9rem);
    letter-spacing: .5px;
    color: #f1f5f9;
    line-height: 1.35;
    margin: .2rem 0 1.1rem;
}

/* Status koneksi */
.sysbar {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: .4rem 1.2rem;
    font-size: .85rem;
    color: #cbd5e1;
    margin-bottom: .8rem;
}

.dot {
    display: inline-block;
    width: .62rem;
    height: .62rem;
    border-radius: 50%;
    margin-right: .35rem;
}

/* Kartu monitoring */
.card {
    background: #172236;
    border: 1px solid #2a3750;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

.metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 160px;
    margin-bottom: .6rem;
}

.metric-title {
    color: #e2e8f0;
    font-weight: 600;
    font-size: .95rem;
}

.metric-value {
    font-size: clamp(2.4rem, 5.5vw, 3.5rem);
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    margin: .35rem 0 .5rem;
    text-shadow: 0 0 18px rgba(255,255,255,.22);
}

.unit {
    font-size: .55em;
    font-weight: 700;
}

.badge {
    display: inline-block;
    padding: .12rem .7rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: .78rem;
    color: #0b1220;
}

.note {
    color: #cbd5e1;
    font-size: .85rem;
    margin-top: .55rem;
}

.bigicon {
    padding-left: .8rem;
    line-height: 0;
}

/* Peringatan */
.sec-title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: .95rem;
    margin-bottom: .6rem;
}

.alert-box {
    border-radius: 8px;
    padding: .7rem 1rem;
    font-size: .9rem;
    color: #f8fafc;
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem 1.8rem;
    margin-top: .85rem;
    font-size: .82rem;
    color: #cbd5e1;
}

.legend-note {
    color: #94a3b8;
    font-size: .74rem;
    margin-top: .55rem;
    line-height: 1.5;
}

/* Histori */
.hist-marker {
    display: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) {
    background: #172236;
    border: 1px solid #2a3750 !important;
    border-radius: 12px;
    padding: .5rem .8rem .6rem;
}

/* Tombol */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stFormSubmitButton"] button {
    background: #e5e7eb;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-weight: 600;
    min-height: 0;
    padding: .3rem .9rem;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #ffffff;
    color: #0f172a;
    border-color: #ffffff;
}

div[data-testid="stButton"] button p,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stFormSubmitButton"] button p {
    color: #0f172a;
    font-size: .82rem;
    font-weight: 600;
}

/* Tabel */
div[data-testid="stDataFrame"] {
    border: 1px solid #2a3750;
    border-radius: 8px;
    overflow: hidden;
}

/* Footer */
.foot {
    color: #64748b;
    font-size: .75rem;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
"""


# ------------------------------------------------------------------
# Utilitas
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Koneksi MQTT
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Menghubungkan ke broker MQTT…")
def get_bridge() -> MqttBridge:
    bridge = MqttBridge(get_settings())
    bridge.wait_ready(4.0)
    return bridge


# ------------------------------------------------------------------
# Koneksi Supabase
# ------------------------------------------------------------------

@st.cache_resource
def get_db():
    s = get_settings()

    if not s.db_configured:
        return None

    return SupabaseClient(
        s.supabase_url,
        s.supabase_key,
        s.timezone
    )


# ------------------------------------------------------------------
# Ikon SVG
# ------------------------------------------------------------------

def _icon_thermo(color: str, value) -> str:
    """Termometer dengan tinggi cairan mengikuti nilai suhu."""

    frac = 0.0 if value is None else min(max(value / 40.0, 0.0), 1.0)
    level_y = 56 - frac * 34

    ticks = "".join(
        f'<line x1="48" y1="{y}" x2="{x2}" y2="{y}" '
        f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        for y, x2 in ((22, 56), (30, 53), (38, 56), (46, 53))
    )

    return (
        f'<svg width="64" height="92" viewBox="0 0 64 92" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="filter:drop-shadow(0 0 8px {_rgba(color, .6)})">'
        f'<path d="M22 16a10 10 0 0 1 20 0V57a18 18 0 1 1-20 0Z" '
        f'stroke="{color}" stroke-width="3" fill="{_rgba(color, .1)}"/>'
        f'<line x1="32" y1="62" x2="32" y2="{level_y:.1f}" '
        f'stroke="{color}" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="32" cy="72" r="10" fill="{color}"/>'
        f'{ticks}</svg>'
    )


def _icon_drop(color: str) -> str:
    return (
        f'<svg width="64" height="84" viewBox="0 0 64 84" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="filter:drop-shadow(0 0 8px {_rgba(color, .6)})">'
        f'<path d="M32 4C32 4 8 34 8 54a24 24 0 0 0 48 0C56 34 32 4 32 4Z" '
        f'fill="{color}"/>'
        f'<path d="M20 56a12 12 0 0 0 9 11" stroke="#fff" '
        f'stroke-opacity=".45" stroke-width="4" stroke-linecap="round"/>'
        f'</svg>'
    )


# ------------------------------------------------------------------
# Komponen kartu monitoring
# ------------------------------------------------------------------

def _metric_card(
    icon: str,
    title: str,
    value,
    unit: str,
    stt: alerts.Status,
    art: str
) -> str:
    c = stt.color

    return _html(f"""
    <div class="card metric"
         style="border-color:{_rgba(c, .75)};
         box-shadow:0 0 22px {_rgba(c, .2)},inset 0 0 30px {_rgba(c, .05)};">

        <div>
            <div class="metric-title">{icon} {title}</div>

            <div class="metric-value">
                {_fmt(value)}
                <span class="unit">{unit if value is not None else ''}</span>
            </div>

            <span class="badge" style="background:{c}">
                ● {stt.level}
            </span>

            <div class="note">{stt.note}</div>
        </div>

        <div class="bigicon">{art}</div>
    </div>
    """)


# ------------------------------------------------------------------
# Legenda peringatan
# ------------------------------------------------------------------

def _legend(l: Limits) -> str:
    items = [
        (
            GREEN,
            f"{l.temp_ideal_min:g} – {l.temp_ideal_max:g} °C",
            "NORMAL"
        ),
        (
            YELLOW,
            f"&gt; {l.temp_ideal_max:g} – {l.temp_crit_max:g} °C",
            "WARNING"
        ),
        (
            RED,
            f"&gt; {l.temp_crit_max:g} °C",
            "MASALAH / KRITIS"
        ),
    ]

    row = '<div class="legend">' + "".join(
        f'<div>{_dot(c)}{rng} : <b>{name}</b></div>'
        for c, rng, name in items
    ) + "</div>"

    note = (
        f'<div class="legend-note">'
        f'Di luar rentang ideal dianggap WARNING '
        f'(termasuk suhu &lt; {l.temp_ideal_min:g} °C); '
        f'suhu kritis bila &lt; {l.temp_crit_min:g} °C. '
        f'Kelembapan: ideal {l.hum_ideal_min:g}–'
        f'{l.hum_ideal_max:g} %RH, kritis bila '
        f'&lt; {l.hum_crit_min:g} atau '
        f'&gt; {l.hum_crit_max:g} %RH.'
        f'</div>'
    )

    return row + note


# ------------------------------------------------------------------
# Monitoring realtime
# ------------------------------------------------------------------

@st.fragment(run_every=2)
def live_panel() -> None:
    s = get_settings()
    bridge = get_bridge()
    snap = bridge.snapshot()

    cfg_dict = snap["config"]

    cfg = (
        from_payload(cfg_dict)
        if cfg_dict
        else ControlConfig()
    )

    limits = cfg.limits

    tel = snap["telemetry"]

    online = tel is not None

    sensor_ok = (
        bool(tel.get("sensor_ok", False))
        if tel else False
    )

    temp = _num(tel.get("temp")) if tel else None
    hum = _num(tel.get("hum")) if tel else None

    # Status sistem
    now = datetime.now(ZoneInfo(s.timezone))

    esp_c, esp_t = (
        (GREEN, "Online")
        if online
        else (RED, "Offline")
    )

    if not online:
        dht_c, dht_t = GREY, "Tidak diketahui"
    elif sensor_ok:
        dht_c, dht_t = GREEN, "Terkoneksi (Online)"
    else:
        dht_c, dht_t = RED, "Error"

    broker_chip = (
        ""
        if snap["broker_connected"]
        else (
            f'<span style="margin-right:1.1rem">'
            f'{_dot(RED)}Broker: Terputus</span>'
        )
    )

    st.markdown(
        _html(f"""
        <div class="sysbar">
            <div>
                Waktu Sistem:
                <b>{now.day} {MONTHS[now.month - 1]} {now.year}</b>
                | {now:%H:%M} {now.tzname()}
            </div>

            <div>
                {broker_chip}

                <span style="margin-right:1.1rem">
                    {_dot(esp_c)}
                    ESP32:
                    <span style="color:{esp_c}">{esp_t}</span>
                </span>

                Status Sensor DHT:
                {_dot(dht_c)}
                <span style="color:{dht_c}">{dht_t}</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    if snap["last_error"]:
        st.warning(snap["last_error"])

    # Kartu temperatur dan kelembapan
    t_st = alerts.temperature_status(
        temp, limits, online, sensor_ok
    )

    h_st = alerts.humidity_status(
        hum, limits, online, sensor_ok
    )

    c1, c2 = st.columns(2)

    c1.markdown(
        _metric_card(
            "🌡️",
            "TEMPERATUR RUANG",
            temp,
            "°C",
            t_st,
            _icon_thermo(t_st.color, temp)
        ),
        unsafe_allow_html=True
    )

    c2.markdown(
        _metric_card(
            "💧",
            "KELEMBAPAN RUANG",
            hum,
            "%RH",
            h_st,
            _icon_drop(h_st.color)
        ),
        unsafe_allow_html=True
    )

    # Sistem peringatan
    pid_active = bool(
        tel
        and tel.get("mode") == "AUTO"
        and tel.get("power")
    )

    level, message = alerts.overall(
        t_st,
        h_st,
        temp,
        hum,
        limits,
        pid_active
    )

    ac = alerts.COLORS[level]

    st.markdown(
        _html(f"""
        <div class="card" style="margin-bottom:.8rem">

            <div class="sec-title">
                🚨 SISTEM PERINGATAN (ALERT)
            </div>

            <div class="alert-box"
                 style="background:{_rgba(ac, .16)};
                 border:1px solid {_rgba(ac, .7)}">

                {_dot(ac)}
                <b>{level}:</b> {message}

            </div>

            {_legend(limits)}

        </div>
        """),
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------
# Memuat histori dari Supabase
# ------------------------------------------------------------------

@st.cache_data(
    ttl=600,
    show_spinner=False,
    max_entries=16
)
def load_history(hours, nonce: int) -> pd.DataFrame:
    # nonce digunakan untuk memaksa pemuatan ulang data
    return get_db().fetch_history(hours)


# ------------------------------------------------------------------
# Memutus garis grafik pada celah data
# ------------------------------------------------------------------

def _with_gaps(
    df: pd.DataFrame,
    gap_minutes: float
) -> pd.DataFrame:

    if len(df) < 2:
        return df

    d = (
        df.sort_values("created_at")
        .reset_index(drop=True)
    )

    delta = d["created_at"].diff()
    gaps = delta > pd.Timedelta(minutes=gap_minutes)

    if not gaps.any():
        return d

    filler = d.loc[gaps].copy()

    filler["created_at"] = (
        d["created_at"].shift(1)[gaps]
        + delta[gaps] / 2
    )

    for c in NUMERIC_COLS:
        filler[c] = np.nan

    return (
        pd.concat([d, filler])
        .sort_values("created_at")
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# Grafik histori
# ------------------------------------------------------------------

def _chart(
    df: pd.DataFrame,
    gap_minutes: float,
    hours
) -> go.Figure:

    d = _with_gaps(df, gap_minutes)

    many = len(d) > 1500
    mode = "lines" if many else "lines+markers"

    tickfmt = (
        "%H:%M"
        if hours is not None and hours <= 24
        else "%d/%m %H:%M"
    )

    fig = go.Figure()

    # Grafik temperatur
    fig.add_trace(
        go.Scatter(
            x=d["created_at"],
            y=d["temperature"],
            name="Temperatur (°C)",
            mode=mode,
            line=dict(color="#ef4444", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.28)",
            connectgaps=False
        )
    )

    # Grafik kelembapan
    fig.add_trace(
        go.Scatter(
            x=d["created_at"],
            y=d["humidity"],
            name="Kelembapan (%)",
            mode=mode,
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)",
            connectgaps=False
        )
    )

    # Setpoint tetap tersedia pada grafik apabila terdapat datanya
    if "setpoint" in d.columns and d["setpoint"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=d["created_at"],
                y=d["setpoint"],
                name="Setpoint (°C)",
                mode="lines",
                line=dict(
                    color="#f5b301",
                    width=1.5,
                    dash="dash"
                ),
                connectgaps=False
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            tickformat=tickfmt
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            rangemode="tozero"
        )
    )

    return fig


# ------------------------------------------------------------------
# Panel histori data
# ------------------------------------------------------------------

@st.fragment(run_every=30)
def history_panel() -> None:
    s = get_settings()

    st.session_state.setdefault("hist_nonce", 0)
    st.session_state.setdefault("show_table", False)

    with st.container(border=True):

        st.markdown(
            '<span class="hist-marker"></span>',
            unsafe_allow_html=True
        )

        title_ph = st.empty()

        c1, c2, c3 = st.columns([3.4, 1.5, 1.1])

        label = c1.radio(
            "Rentang waktu",
            list(RANGES),
            index=3,
            horizontal=True,
            key="hist_range",
            label_visibility="collapsed"
        )

        auto = c2.toggle(
            "Auto-refresh grafik",
            value=True,
            key="hist_auto"
        )

        if c3.button("↻ Muat ulang", key="hist_reload"):
            st.session_state.hist_nonce += 1

        hours = RANGES[label]

        title_ph.markdown(
            f'<div class="sec-title">'
            f'📈 HISTORI DATA ('
            f'{"REKAPAN " + label.upper() + " TERAKHIR" if hours else "SELURUH HISTORI"}'
            f')</div>',
            unsafe_allow_html=True
        )

        # Periksa konfigurasi database
        if get_db() is None:
            st.warning(
                "Database belum dikonfigurasi "
                "(SUPABASE_URL / SUPABASE_SERVICE_KEY)."
            )
            return

        nonce = (
            int(time.time() // 30)
            if auto
            else st.session_state.hist_nonce
        )

        try:
            df = load_history(hours, nonce)

        except Exception as exc:
            st.error(
                f"Gagal membaca histori dari Supabase: {exc}"
            )
            return

        if df.empty:
            st.info(
                "Belum ada data tersimpan pada rentang ini. "
                "Pastikan ESP32 online dan tabel `sensor_log` "
                "di Supabase sudah dibuat "
                "(lihat sql/schema.sql)."
            )
            return

        # Grafik histori
        st.plotly_chart(
            _chart(df, s.gap_minutes, hours),
            config={"displayModeBar": False},
            use_container_width=True
        )

        # Ekspor CSV tetap menyimpan seluruh kolom data
        out = df.copy()

        out["created_at"] = (
            out["created_at"]
            .dt.strftime("%Y-%m-%d %H:%M:%S%z")
        )

        _, b1, b2 = st.columns([2.6, 1.5, 1.7])

        b1.download_button(
            "Data Ekspor (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"histori_arsip_"
                f"{datetime.now():%Y%m%d_%H%M}.csv"
            ),
            mime="text/csv",
            key="dl_csv"
        )

        table_label = (
            "Sembunyikan Tabel Log"
            if st.session_state.show_table
            else "Tampilkan Tabel Log"
        )

        if b2.button(table_label, key="btn_table"):
            st.session_state.show_table = (
                not st.session_state.show_table
            )
            st.rerun(scope="fragment")

        st.caption(
            f"{len(df):,} pembacaan · terakhir: "
            f"{df['created_at'].iloc[-1]:%d-%m-%Y %H:%M:%S}"
        )

        # Tabel log hanya menampilkan tiga kolom monitoring
        if st.session_state.show_table:

            visible_cols = [
                "created_at",
                "temperature",
                "humidity"
            ]

            # Pastikan hanya kolom yang tersedia yang dipilih
            visible_cols = [
                col for col in visible_cols
                if col in df.columns
            ]

            view = (
                df.sort_values(
                    "created_at",
                    ascending=False
                )
                .head(1000)[visible_cols]
                .copy()
            )

            view["created_at"] = (
                view["created_at"]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

            st.dataframe(
                view.rename(columns=TABLE_COLS),
                hide_index=True,
                use_container_width=True
            )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def render() -> None:
    s = get_settings()

    st.markdown(
        CSS,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="title">'
        'SISTEM KENDALI PID IKLIM MIKRO RUANG ARSIP BERBASIS IoT<br>'
        'DENGAN ANTARMUKA PEMANTAUAN REAL-TIME'
        '</div>',
        unsafe_allow_html=True
    )

    # Pemeriksaan konfigurasi MQTT
    if not s.mqtt_configured:
        st.error(
            "Kredensial MQTT belum diisi. Salin "
            "`.streamlit/secrets.toml.example` menjadi "
            "`.streamlit/secrets.toml` lalu isi MQTT_HOST, "
            "MQTT_USERNAME, dan MQTT_PASSWORD."
        )
        st.stop()

    # Monitoring realtime
    live_panel()

    # Histori data
    history_panel()

    # Footer
    st.markdown(
        '<div class="foot">'
        'Dashboard monitoring iklim mikro ruang arsip '
        '— algoritma PID berjalan di ESP32.'
        '</div>',
        unsafe_allow_html=True
    )
