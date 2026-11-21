"""Jembatan MQTT antara Streamlit dan ESP32.

- Subscribe : <prefix>/telemetry (data sensor + status PID dari ESP32)
              <prefix>/status    (online/offline; 'offline' dikirim broker via LWT)
              <prefix>/config    (konfigurasi retained, sumber kebenaran setpoint/PID/batas alert)
- Publish   : <prefix>/config    (retained, QoS 1)

Status ESP32 dinilai ONLINE hanya bila broker terhubung, telemetri terakhir
berumur <= offline_timeout_s, dan status LWT bukan 'offline'. Bila tidak,
telemetri lama TIDAK dikembalikan ke UI.
"""
from __future__ import annotations

import json
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from config import Settings
from pid_control import ControlConfig, to_payload, validate


class MqttBridge:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self._telemetry: dict | None = None
        self._telemetry_rx = 0.0
        self._status_msg = "unknown"
        self._config: dict | None = None
        self._connected = False
        self._last_error = ""
        self._config_event = threading.Event()

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"dash-{uuid.uuid4().hex[:10]}",
        )
        self._client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
        if settings.mqtt_port in (8883, 8884):
            self._client.tls_set()  # memakai CA sistem
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.connect_async(settings.mqtt_host, settings.mqtt_port, keepalive=30)
        self._client.loop_start()

    # ---- callback paho (thread jaringan) ----
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        failed = getattr(reason_code, "is_failure", reason_code != 0)
        with self._lock:
            self._connected = not failed
            self._last_error = f"Gagal terhubung ke broker: {reason_code}" if failed else ""
        if not failed:
            client.subscribe([(self.s.topic_telemetry, 1),
                              (self.s.topic_status, 1),
                              (self.s.topic_config, 1)])

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        with self._lock:
            self._connected = False

    def _on_message(self, client, userdata, msg):
        raw = msg.payload.decode("utf-8", "ignore").strip()
        if msg.topic == self.s.topic_status:
            with self._lock:
                self._status_msg = raw.lower()
            return
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        if msg.topic == self.s.topic_telemetry:
            with self._lock:
                self._telemetry = data
                self._telemetry_rx = time.time()
        elif msg.topic == self.s.topic_config:
            with self._lock:
                self._config = data
            self._config_event.set()

    # ---- API untuk UI ----
    def wait_ready(self, timeout: float = 4.0) -> None:
        """Tunggu singkat sampai konfigurasi retained diterima (sekali saat start)."""
        self._config_event.wait(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            age = (time.time() - self._telemetry_rx) if self._telemetry_rx else None
            online = bool(
                self._connected
                and age is not None
                and age <= self.s.offline_timeout_s
                and self._status_msg != "offline"
            )
            return {
                "broker_connected": self._connected,
                "esp_online": online,
                "telemetry_age_s": age,
                "telemetry": dict(self._telemetry) if (online and self._telemetry) else None,
                "config": dict(self._config) if self._config else None,
                "last_error": self._last_error,
            }

    def publish_config(self, cfg: ControlConfig) -> tuple[bool, str]:
        errs = validate(cfg)
        if errs:
            return False, " ".join(errs)
        with self._lock:
            connected = self._connected
        if not connected:
            return False, "Dashboard belum terhubung ke broker MQTT."
        payload = to_payload(cfg)
        try:
            info = self._client.publish(self.s.topic_config, json.dumps(payload), qos=1, retain=True)
            info.wait_for_publish(timeout=5)
        except Exception as exc:  # noqa: BLE001
            return False, f"Gagal mengirim ke broker: {exc}"
        if not info.is_published():
            return False, "Broker tidak mengonfirmasi pengiriman (timeout)."
        with self._lock:
            self._config = payload
        return True, "Konfigurasi terkirim ke broker (retained)."
