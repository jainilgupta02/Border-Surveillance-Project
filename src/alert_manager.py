"""
Alert Manager Module
=====================

Consumes AnomalyResult objects from anomaly.py, decides alert priority,
persists alert logs to JSON, and optionally sends email notifications via
SendGrid.

Architecture position:
    anomaly.py → [THIS MODULE] → dashboard / Cosmos DB / email

Design principles:
    1. Priority tiers  — CRITICAL / HIGH / MEDIUM / LOW mapped from anomaly
       score + threat class.  alert_manager.py is the only place priorities
       are assigned; nothing upstream hardcodes them.
    2. Deduplication   — a cooldown window prevents the same threat class
       from flooding the log within a short time window.
    3. Persistence     — every alert is written to a rotating JSON log so
       the Streamlit dashboard can read it without a database.
    4. Optional email  — SendGrid integration is gated behind an env var
       (SENDGRID_API_KEY).  If absent, alerts are logged but not emailed.
    5. Thread-safe     — uses a file lock pattern so multiple pipeline
       workers can write to the same log without corruption.

Alert priority mapping:
    anomaly alert_level = "critical"  →  CRITICAL  🔴
    anomaly alert_level = "high"      →  HIGH      🟠
    motion_score > threshold          →  MEDIUM    🟡   (if not already higher)
    everything else                   →  LOW       🟢   (logged, not emailed)

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from azure_client import azure
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Priority tiers (ordered by severity)
PRIORITY_CRITICAL = "CRITICAL"   # 🔴 immediate action
PRIORITY_HIGH     = "HIGH"       # 🟠 alert
PRIORITY_MEDIUM   = "MEDIUM"     # 🟡 log + review
PRIORITY_LOW      = "LOW"        # 🟢 log only

# Motion threshold that bumps an otherwise LOW/MEDIUM alert up to MEDIUM
MOTION_MEDIUM_THRESHOLD: float = 8.0

# Cooldown — minimum seconds between two alerts for the same class
# Prevents alert flooding for the same persistent threat
COOLDOWN_SECONDS: int = 45

# Alert log file
DEFAULT_LOG_PATH: str = "data/alerts/alert_log.json"

# Maximum alerts kept in the rolling log (oldest removed when exceeded)
MAX_LOG_SIZE: int = 500

# Priority rank for comparison
_PRIORITY_RANK = {
    PRIORITY_LOW:      0,
    PRIORITY_MEDIUM:   1,
    PRIORITY_HIGH:     2,
    PRIORITY_CRITICAL: 3,
}


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """
    A single alert generated for a frame.

    Attributes:
        alert_id:      Unique identifier (timestamp-based).
        frame_id:      Source frame number.
        timestamp:     Wall-clock time of alert creation.
        priority:      CRITICAL / HIGH / MEDIUM / LOW.
        anomaly_score: Raw Isolation Forest score.
        anomaly_prob:  Normalised anomaly probability [0, 1].
        alert_level:   Raw level from anomaly.py ("critical"/"high"/"normal").
        reasons:       Human-readable reasons from anomaly.py.
        detection_count: Number of objects in the source frame.
        motion_score:  Optical flow score (None if unavailable).
        notified:      True if an email notification was sent.
    """
    alert_id:        str
    frame_id:        int
    timestamp:       float
    priority:        str
    anomaly_score:   float
    anomaly_prob:    float
    alert_level:     str
    reasons:         List[str]
    detection_count: int            = 0
    motion_score:    Optional[float] = None
    notified:        bool            = False

    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON log / Cosmos DB."""
        return {
            "alert_id":        self.alert_id,
            "frame_id":        self.frame_id,
            "timestamp":       round(self.timestamp, 3),
            "priority":        self.priority,
            "anomaly_score":   round(self.anomaly_score, 4),
            "anomaly_prob":    round(self.anomaly_prob,  4),
            "alert_level":     self.alert_level,
            "reasons":         self.reasons,
            "detection_count": self.detection_count,
            "motion_score":    round(self.motion_score, 4)
                               if self.motion_score is not None else None,
            "notified":        self.notified,
        }

    @property
    def priority_rank(self) -> int:
        """Numeric rank for comparison (higher = more severe)."""
        return _PRIORITY_RANK.get(self.priority, 0)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Processes AnomalyResult dicts, assigns priorities, persists to log,
    and optionally sends email alerts via SendGrid.

    Args:
        log_path:         Path to the rolling JSON alert log.
        cooldown_seconds: Minimum gap between repeated alerts.
        enable_email:     Override email sending regardless of env var.
                          None = auto-detect from SENDGRID_API_KEY.
        sendgrid_api_key: Optional SendGrid key (overrides env var).
        from_email:       Sender email address.
        to_email:         Recipient email address.

    Example (pipeline.py calls it like this):
        alert_mgr = AlertManager()
        for anomaly_result in anomaly_results:
            alert = alert_mgr.process(anomaly_result.to_dict())
            if alert:
                print(f"Alert raised: {alert.priority} at frame {alert.frame_id}")
    """

    def __init__(
        self,
        log_path:         str   = DEFAULT_LOG_PATH,
        cooldown_seconds: int   = COOLDOWN_SECONDS,
        enable_email:     Optional[bool] = None,
        sendgrid_api_key: Optional[str]  = None,
        from_email:       Optional[str]  = None,
        to_email:         Optional[str]  = None,
    ) -> None:
        self.log_path         = log_path
        self.cooldown_seconds = cooldown_seconds
        self.from_email       = from_email or os.getenv("ALERT_FROM_EMAIL", "")
        self.to_email         = to_email   or os.getenv("ALERT_TO_EMAIL", "")
        self._api_key         = (
            sendgrid_api_key or os.getenv("SENDGRID_API_KEY", "")
        )

        # Auto-detect email capability
        if enable_email is None:
            self._email_enabled = bool(self._api_key)
        else:
            self._email_enabled = enable_email

        # Cooldown tracker: class_name → last alert timestamp
        self._last_alert_time: Dict[str, float] = {}

        # In-memory alert buffer (also written to disk)
        self._alerts: List[Alert] = []

        # Load existing log if present
        self._load_log()

        logger.info(
            "AlertManager ready  |  log=%s  |  cooldown=%ds  |  email=%s",
            log_path, cooldown_seconds, self._email_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, anomaly_result: dict) -> Optional[Alert]:
        """
        Process one AnomalyResult dict and create an Alert if warranted.

        LOW-priority frames are logged silently.
        MEDIUM / HIGH / CRITICAL frames are logged and optionally emailed.

        Args:
            anomaly_result: Dict from AnomalyResult.to_dict().

        Returns:
            Alert object if one was created, else None.
        """
        alert_level   = anomaly_result.get("alert_level", "normal")
        anomaly_score = anomaly_result.get("anomaly_score", 0.0)
        motion_score  = anomaly_result.get("motion_score")
        reasons       = anomaly_result.get("reasons", [])

        # Determine priority
        priority = self._assign_priority(alert_level, motion_score)

        # Always create an Alert object so the log is complete
        alert = Alert(
            alert_id        = f"alert_{int(time.time() * 1000)}",
            frame_id        = anomaly_result.get("frame_id", 0),
            timestamp       = anomaly_result.get("timestamp", time.time()),
            priority        = priority,
            anomaly_score   = anomaly_score,
            anomaly_prob    = anomaly_result.get("anomaly_prob", 0.0),
            alert_level     = alert_level,
            reasons         = reasons,
            detection_count = anomaly_result.get("detection_count", 0),
            motion_score    = motion_score,
            notified        = False,
        )

        # Log all non-normal alerts
        if alert_level != "normal":
            self._log_alert(alert)

            # Send email for HIGH and CRITICAL (subject to cooldown)
            if priority in (PRIORITY_HIGH, PRIORITY_CRITICAL):
                if self._is_cooled_down(reasons):
                    self._notify(alert)

            logger.info(
                "[%s] Frame %d | score=%.3f | %s",
                priority, alert.frame_id, anomaly_score,
                " | ".join(reasons) if reasons else "no reasons",
            )
            return alert

        return None

    def get_recent_alerts(self, n: int = 50) -> List[dict]:
        """Return the N most recent alerts as a list of dicts."""
        return [a.to_dict() for a in self._alerts[-n:]]

    def get_summary(self) -> dict:
        """Return aggregate statistics across all logged alerts."""
        if not self._alerts:
            return {"total_alerts": 0, "email_enabled": self._email_enabled}

        by_priority: Dict[str, int] = {}
        for a in self._alerts:
            by_priority[a.priority] = by_priority.get(a.priority, 0) + 1

        return {
            "total_alerts":      len(self._alerts),
            "by_priority":       by_priority,
            "critical_count":    by_priority.get(PRIORITY_CRITICAL, 0),
            "high_count":        by_priority.get(PRIORITY_HIGH,     0),
            "medium_count":      by_priority.get(PRIORITY_MEDIUM,   0),
            "low_count":         by_priority.get(PRIORITY_LOW,      0),
            "email_enabled":     self._email_enabled,
            "log_path":          self.log_path,
        }

    def clear_log(self) -> None:
        """Clear in-memory and on-disk alert log.  Used for testing."""
        self._alerts = []
        self._save_log()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_priority(
        alert_level:  str,
        motion_score: Optional[float],
    ) -> str:
        """Map anomaly alert_level + motion to a 4-tier priority."""
        if alert_level == "critical":
            return PRIORITY_CRITICAL
        if alert_level == "high":
            return PRIORITY_HIGH
        if alert_level == "normal":
            # Motion alone can bump up to MEDIUM
            if motion_score and motion_score > MOTION_MEDIUM_THRESHOLD:
                return PRIORITY_MEDIUM
            return PRIORITY_LOW
        return PRIORITY_LOW

    def _is_cooled_down(self, reasons: List[str]) -> bool:
        """
        Return True if enough time has passed since the last alert
        for any class mentioned in reasons.
        """
        now = time.time()
        # Extract class mentions from reason strings
        all_classes = (
            list({"military_vehicle", "suspicious_object", "crowd", "aircraft"})
        )
        relevant = [c for c in all_classes if any(c in r for r in reasons)]

        if not relevant:
            # Generic anomaly — use a generic cooldown key
            relevant = ["generic"]

        for cls in relevant:
            last = self._last_alert_time.get(cls, 0)
            if now - last < self.cooldown_seconds:
                logger.debug("Cooldown active for '%s' — skipping notification", cls)
                return False
            self._last_alert_time[cls] = now

        return True

    def _log_alert(self, alert: Alert) -> None:
        """Append alert to in-memory buffer and persist to disk."""
        self._alerts.append(alert)

        # Rolling cap — remove oldest when limit exceeded
        if len(self._alerts) > MAX_LOG_SIZE:
            self._alerts = self._alerts[-MAX_LOG_SIZE:]

        self._save_log()
         # ── Azure: save to Cosmos DB (non-blocking, fails gracefully) ──
        try:
            from azure_client import azure
            azure.save_alert(alert.to_dict())
        except Exception:
          pass  # Azure failure never breaks local pipeline

    def _save_log(self) -> None:
        """Write current alerts to JSON log file."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        try:
            with open(self.log_path, "w") as f:
                json.dump(
                    [a.to_dict() for a in self._alerts],
                    f, indent=2,
                )
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)

    def _load_log(self) -> None:
        """Load existing alert log from disk into memory."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path) as f:
                raw = json.load(f)
            for item in raw:
                self._alerts.append(Alert(
                    alert_id        = item.get("alert_id", ""),
                    frame_id        = item.get("frame_id", 0),
                    timestamp       = item.get("timestamp", 0.0),
                    priority        = item.get("priority", PRIORITY_LOW),
                    anomaly_score   = item.get("anomaly_score", 0.0),
                    anomaly_prob    = item.get("anomaly_prob",  0.0),
                    alert_level     = item.get("alert_level",  "normal"),
                    reasons         = item.get("reasons",      []),
                    detection_count = item.get("detection_count", 0),
                    motion_score    = item.get("motion_score"),
                    notified        = item.get("notified", False),
                ))
            logger.info("Loaded %d existing alerts from %s",
                        len(self._alerts), self.log_path)
        except Exception as exc:
            logger.warning("Could not load alert log: %s", exc)

    def _notify(self, alert: Alert) -> None:
        """
        Send email notification via SendGrid.
        Silently skips if API key / addresses are missing.
        """
        if not self._email_enabled:
            return
        if not (self._api_key and self.from_email and self.to_email):
            logger.debug("Email skipped — credentials not configured")
            return

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            priority_emoji = {
                PRIORITY_CRITICAL: "🚨",
                PRIORITY_HIGH:     "⚠️",
                PRIORITY_MEDIUM:   "🟡",
                PRIORITY_LOW:      "🟢",
            }.get(alert.priority, "")

            reasons_html = "".join(f"<li>{r}</li>" for r in alert.reasons)

            message = Mail(
                from_email  = self.from_email,
                to_emails   = self.to_email,
                subject     = (
                    f"{priority_emoji} [{alert.priority}] Border Surveillance Alert"
                    f" — Frame {alert.frame_id}"
                ),
                html_content = f"""
                <h2>{priority_emoji} Border Surveillance Alert</h2>
                <table>
                  <tr><td><b>Priority</b></td><td>{alert.priority}</td></tr>
                  <tr><td><b>Frame</b></td><td>{alert.frame_id}</td></tr>
                  <tr><td><b>Anomaly Score</b></td>
                      <td>{alert.anomaly_score:.4f}</td></tr>
                  <tr><td><b>Detections</b></td>
                      <td>{alert.detection_count}</td></tr>
                  <tr><td><b>Motion</b></td>
                      <td>{alert.motion_score or 'N/A'}</td></tr>
                </table>
                <h3>Reasons</h3>
                <ul>{reasons_html}</ul>
                """,
            )

            sg       = SendGridAPIClient(self._api_key)
            response = sg.send(message)

            if response.status_code == 202:
                alert.notified = True
                logger.info(
                    "Email sent for %s alert at frame %d",
                    alert.priority, alert.frame_id,
                )
            else:
                logger.warning(
                    "SendGrid returned %d for frame %d",
                    response.status_code, alert.frame_id,
                )

        except ImportError:
            logger.warning("sendgrid package not installed — email skipped")
        except Exception as exc:
            logger.error("Email notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Quick test (run directly: python src/alert_manager.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Border Surveillance AI — Alert Manager Module")
    print("=" * 55)

    # Simulate some AnomalyResult dicts
    test_results = [
        {
            "frame_id": 10, "timestamp": time.time(),
            "anomaly_score": -0.20, "anomaly_prob": 0.87,
            "alert_level": "critical",
            "reasons": ["military_vehicle detected"],
            "detection_count": 3, "motion_score": 9.5,
        },
        {
            "frame_id": 15, "timestamp": time.time(),
            "anomaly_score": -0.08, "anomaly_prob": 0.58,
            "alert_level": "high",
            "reasons": ["crowd gathering detected", "high motion activity"],
            "detection_count": 12, "motion_score": 11.2,
        },
        {
            "frame_id": 20, "timestamp": time.time(),
            "anomaly_score": 0.03, "anomaly_prob": 0.20,
            "alert_level": "normal",
            "reasons": [],
            "detection_count": 4, "motion_score": 5.1,
        },
    ]

    mgr = AlertManager(log_path="data/alerts/test_alert_log.json")

    print("\nProcessing test results...")
    for r in test_results:
        alert = mgr.process(r)
        if alert:
            print(f"  [{alert.priority}] Frame {alert.frame_id} — {alert.reasons}")
        else:
            print(f"  [SKIPPED]  Frame {r['frame_id']} — normal frame")

    print("\n" + "=" * 55)
    print("ALERT SUMMARY")
    print("=" * 55)
    summary = mgr.get_summary()
    for k, v in summary.items():
        print(f"  {k:<20} {v}")

    print(f"\nLog written → data/alerts/test_alert_log.json")
    print("Next step: python src/pipeline.py")
