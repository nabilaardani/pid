"""Akses histori data sensor di Supabase (PostgREST) menggunakan requests.

Penulisan data dilakukan langsung oleh ESP32 (HTTPS POST).
Modul ini hanya membaca.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

TABLE = "sensor_log"
NUMERIC_COLS = [
    "temperature", "humidity", "setpoint", "error",
    "kp", "ki", "kd", "duty"
]
COLUMNS = ["created_at", *NUMERIC_COLS, "actuator", "mode", "power"]


class SupabaseClient:
    def __init__(
        self,
        url: str,
        key: str,
        tz: str = "Asia/Jakarta",
        timeout: int = 15
    ):
        self.url = url.rstrip("/")
        self.key = key.strip()
        self.tz = tz
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Accept": "application/json",
        }
        # Kunci JWT lama (anon/service_role) juga perlu header Authorization.
        # Kunci baru (sb_secret_...) cukup lewat header apikey.
        if self.key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def fetch_history(
        self,
        hours: float | None = None,
        max_rows: int = 50_000,
        page_size: int = 1000
    ) -> pd.DataFrame:
        """Ambil histori terbaru, lalu kembalikan dalam urutan waktu naik."""
        base = {
            "select": ",".join(COLUMNS),
            "order": "created_at.desc"
        }

        if hours is not None:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            base["created_at"] = (
                "gte." + since.strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        rows: list[dict] = []
        offset = 0

        while len(rows) < max_rows:
            params = {
                **base,
                "limit": min(page_size, max_rows - len(rows)),
                "offset": offset
            }

            response = requests.get(
                f"{self.url}/rest/v1/{TABLE}",
                headers=self._headers(),
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            rows.extend(batch)
            offset += len(batch)

        return self._to_frame(rows)

    def _to_frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=COLUMNS)

        # Parsing per elemen: tahan terhadap jumlah digit pecahan detik
        # yang berbeda-beda (mis. .5738 vs .573848 vs tanpa pecahan).
        timestamps = df["created_at"].map(
            lambda value: pd.Timestamp(value) if value else pd.NaT
        )
        df["created_at"] = pd.to_datetime(
            timestamps,
            utc=True
        ).dt.tz_convert(self.tz)

        for column in NUMERIC_COLS:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        return df.sort_values("created_at").reset_index(drop=True)