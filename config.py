"""Konfigurasi aplikasi.

Semua nilai sensitif dibaca dari Streamlit Secrets (.streamlit/secrets.toml,
atau menu Secrets di Streamlit Community Cloud) atau dari environment variable.
Tidak ada kredensial yang ditulis langsung di dalam kode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    topic_prefix: str
    supabase_url: str
    supabase_key: str
    dashboard_password: str
    timezone: str
    offline_timeout_s: float   # ESP32 dianggap offline jika tidak ada telemetri selama ini
    gap_minutes: float         # celah data > ini digambar putus pada grafik

    @property
    def topic_telemetry(self) -> str:
        return f"{self.topic_prefix}/telemetry"

    @property
    def topic_status(self) -> str:
        return f"{self.topic_prefix}/status"

    @property
    def topic_config(self) -> str:
        return f"{self.topic_prefix}/config"

    @property
    def mqtt_configured(self) -> bool:
        return bool(self.mqtt_host and self.mqtt_username and self.mqtt_password)

    @property
    def db_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


def get_settings() -> Settings:
    return Settings(
        mqtt_host=_get("MQTT_HOST"),
        mqtt_port=int(_get("MQTT_PORT", "8883")),
        mqtt_username=_get("MQTT_USERNAME"),
        mqtt_password=_get("MQTT_PASSWORD"),
        topic_prefix=_get("MQTT_TOPIC_PREFIX", "arsip").strip("/"),
        supabase_url=_get("SUPABASE_URL").rstrip("/"),
        supabase_key=_get("SUPABASE_SERVICE_KEY"),
        dashboard_password=_get("DASHBOARD_PASSWORD"),
        timezone=_get("APP_TIMEZONE", "Asia/Jakarta"),
        offline_timeout_s=float(_get("ESP_OFFLINE_TIMEOUT_S", "20")),
        gap_minutes=float(_get("CHART_GAP_MINUTES", "5")),
    )
