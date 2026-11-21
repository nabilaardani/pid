"""Jembatan MQTT antara Streamlit dan ESP32.

Subscribe:
- <prefix>/telemetry : data sensor dan status PID dari ESP32
- <prefix>/status    : status online/offline dari ESP32 (LWT)
- <prefix>/config    : konfigurasi retained

Publish:
- <prefix>/config    : konfigurasi dari dashboard (retained, QoS 1)

Status ESP32 dianggap ONLINE hanya jika:
1. Dashboard terhubung ke broker MQTT.
2. Telemetri terbaru masih berada dalam batas offline_timeout_s.
3. Status LWT bukan "offline".

Telemetri lama tidak ditampilkan setelah koneksi terputus atau saat menunggu
telemetri baru setelah broker tersambung kembali.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

import paho.mqtt.client as mqtt

from config import Settings
from pid_control import ControlConfig, to_payload, validate


class MqttBridge:
    """Mengelola koneksi MQTT dashboard dengan callback yang aman untuk thread."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.RLock()

        self._telemetry: dict[str, Any] | None = None
        self._telemetry_rx = 0.0
        self._status_msg = "unknown"
        self._config: dict[str, Any] | None = None

        self._connected = False
        self._last_error = ""
        self._last_error_at = 0.0
        self._config_event = threading.Event()

        # Validasi konfigurasi sebelum memulai thread jaringan.
        missing = []
        if not settings.mqtt_host:
            missing.append("MQTT_HOST")
        if not settings.mqtt_username:
            missing.append("MQTT_USERNAME")
        if not settings.mqtt_password:
            missing.append("MQTT_PASSWORD")

        if missing:
            self._set_error(
                "Konfigurasi MQTT belum lengkap: " + ", ".join(missing)
            )
            return

        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"dash-{uuid.uuid4().hex[:10]}",
                protocol=mqtt.MQTTv311,
            )
            self._client.username_pw_set(
                settings.mqtt_username,
                settings.mqtt_password,
            )

            # EMQX TLS umumnya menggunakan port 8883.
            if int(settings.mqtt_port) in (8883, 8884):
                self._client.tls_set()

            self._client.on_connect = self._on_connect
            self._client.on_connect_fail = self._on_connect_fail
            self._client.on_disconnect = self._on_disconnect
            self._client.on_subscribe = self._on_subscribe
            self._client.on_message = self._on_message
            self._client.on_log = self._on_log

            self._client.reconnect_delay_set(min_delay=1, max_delay=30)

            self._client.connect_async(
                settings.mqtt_host,
                int(settings.mqtt_port),
                keepalive=30,
            )
            self._client.loop_start()

        except Exception as exc:  # noqa: BLE001
            self._set_error(
                f"Gagal memulai koneksi MQTT: {type(exc).__name__}: {exc}"
            )

    # ---------------------------------------------------------- error helper
    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message)
            self._last_error_at = time.time()

    @staticmethod
    def _reason_failed(reason_code: Any) -> bool:
        """Kompatibel dengan ReasonCode Paho v2 dan nilai integer."""
        failure = getattr(reason_code, "is_failure", None)
        if failure is not None:
            return bool(failure)
        try:
            return int(reason_code) != 0
        except (TypeError, ValueError):
            return str(reason_code).lower() not in ("0", "success", "normal")

    # -------------------------------------------------------- Paho callbacks
    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ) -> None:
        failed = self._reason_failed(reason_code)

        with self._lock:
            self._connected = not failed

            if failed:
                self._last_error = (
                    f"Broker menolak koneksi MQTT. Kode alasan: {reason_code}. "
                    "Periksa username, password, dan aturan autentikasi EMQX."
                )
                self._last_error_at = time.time()
                self._telemetry = None
                self._telemetry_rx = 0.0
                return

            self._last_error = ""
            self._last_error_at = 0.0

            # Wajib menerima telemetri baru setelah reconnect.
            self._telemetry = None
            self._telemetry_rx = 0.0
            self._status_msg = "unknown"

        topics = [
            (self.s.topic_telemetry, 1),
            (self.s.topic_status, 1),
            (self.s.topic_config, 1),
        ]

        try:
            result, _mid = client.subscribe(topics)
            if result != mqtt.MQTT_ERR_SUCCESS:
                self._set_error(
                    f"Koneksi broker berhasil, tetapi permintaan subscribe "
                    f"gagal. Kode Paho: {result}."
                )
        except Exception as exc:  # noqa: BLE001
            self._set_error(
                f"Gagal melakukan subscribe MQTT: {type(exc).__name__}: {exc}"
            )

    def _on_connect_fail(self, client, userdata) -> None:
        self._set_error(
            "Gagal menjangkau broker MQTT. Periksa alamat host, port, "
            "koneksi internet, dan pengaturan TLS."
        )

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags=None,
        reason_code=None,
        properties=None,
    ) -> None:
        with self._lock:
            self._connected = False
            self._telemetry = None
            self._telemetry_rx = 0.0

        if reason_code is not None and self._reason_failed(reason_code):
            self._set_error(
                f"Koneksi MQTT terputus. Kode alasan: {reason_code}. "
                "Periksa kestabilan jaringan dan konfigurasi broker."
            )
        else:
            # Jangan menampilkan error untuk pemutusan normal.
            with self._lock:
                self._status_msg = "unknown"

    def _on_subscribe(
        self,
        client,
        userdata,
        mid,
        reason_code_list,
        properties=None,
    ) -> None:
        rejected = [
            str(code)
            for code in (reason_code_list or [])
            if self._reason_failed(code)
        ]
        if rejected:
            self._set_error(
                "Broker menolak subscribe ke salah satu topic. "
                f"Kode alasan: {', '.join(rejected)}. "
                "Periksa ACL/permission akun dashboard pada EMQX "
                "untuk topic arsip/#."
            )

    def _on_log(self, client, userdata, level, buf) -> None:
        # Hanya simpan pesan warning/error agar UI tidak dipenuhi log rutin.
        if level in (mqtt.MQTT_LOG_ERR, mqtt.MQTT_LOG_WARNING):
            message = str(buf).strip()
            if message:
                self._set_error(f"Log MQTT: {message}")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            raw = msg.payload.decode("utf-8", "strict").strip()
        except UnicodeDecodeError:
            self._set_error(
                f"Payload pada topic {msg.topic} bukan teks UTF-8 yang valid."
            )
            return

        if msg.topic == self.s.topic_status:
            with self._lock:
                self._status_msg = raw.lower()
            return

        if not raw:
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._set_error(
                f"Payload JSON tidak valid pada topic {msg.topic}: {exc.msg}."
            )
            return

        if not isinstance(data, dict):
            self._set_error(
                f"Format payload pada topic {msg.topic} harus berupa objek JSON."
            )
            return

        if msg.topic == self.s.topic_telemetry:
            with self._lock:
                self._telemetry = data
                self._telemetry_rx = time.time()
                # Pesan error koneksi/subscribe tidak lagi relevan setelah
                # telemetri berhasil diterima.
                if self._connected:
                    self._last_error = ""
                    self._last_error_at = 0.0

        elif msg.topic == self.s.topic_config:
            with self._lock:
                self._config = data
            self._config_event.set()

    # --------------------------------------------------------------- API UI
    def wait_ready(self, timeout: float = 4.0) -> None:
        """Tunggu singkat sampai konfigurasi retained diterima."""
        self._config_event.wait(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age = (
                time.time() - self._telemetry_rx
                if self._telemetry_rx
                else None
            )

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
                "telemetry": (
                    dict(self._telemetry)
                    if online and self._telemetry is not None
                    else None
                ),
                "config": (
                    dict(self._config)
                    if self._config is not None
                    else None
                ),
                "last_error": self._last_error,
            }

    def publish_config(self, cfg: ControlConfig) -> tuple[bool, str]:
        """Validasi lalu kirim konfigurasi retained ke ESP32."""
        errors = validate(cfg)
        if errors:
            return False, " ".join(errors)

        with self._lock:
            connected = self._connected

        if not connected:
            snap = self.snapshot()
            detail = snap.get("last_error") or (
                "Dashboard belum terhubung ke broker MQTT."
            )
            return False, detail

        payload = to_payload(cfg)

        try:
            info = self._client.publish(
                self.s.topic_config,
                json.dumps(payload),
                qos=1,
                retain=True,
            )

            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                return (
                    False,
                    f"Gagal mengirim konfigurasi. Kode Paho: {info.rc}.",
                )

            info.wait_for_publish(timeout=5)

            if not info.is_published():
                return (
                    False,
                    "Broker belum mengonfirmasi pengiriman konfigurasi "
                    "setelah 5 detik (timeout).",
                )

        except Exception as exc:  # noqa: BLE001
            message = (
                f"Gagal mengirim konfigurasi MQTT: "
                f"{type(exc).__name__}: {exc}"
            )
            self._set_error(message)
            return False, message

        with self._lock:
            self._config = payload

        return True, "Konfigurasi terkirim ke broker (retained)."

    def close(self) -> None:
        """Hentikan koneksi dengan aman jika bridge perlu ditutup."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.disconnect()
            client.loop_stop()
        except Exception as exc:  # noqa: BLE001
            self._set_error(
                f"Gagal menghentikan koneksi MQTT: {type(exc).__name__}: {exc}"
            )
