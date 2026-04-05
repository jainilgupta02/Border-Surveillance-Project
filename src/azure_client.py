"""
Azure Integration Module
========================
Handles Blob Storage uploads and Cosmos DB writes.
Gracefully degrades if Azure is not configured — local pipeline
continues working even if Azure credentials are missing.

Author: Border Surveillance AI (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy imports so pipeline works without Azure SDK installed ────────────
try:
    from azure.storage.blob import BlobServiceClient
    from azure.cosmos import CosmosClient
    from dotenv import load_dotenv
    load_dotenv()
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure SDK not installed — cloud upload disabled. "
                   "pip install azure-storage-blob azure-cosmos python-dotenv")


class AzureClient:
    """
    Thin wrapper around Blob Storage + Cosmos DB.
    All methods are safe to call even if Azure is unconfigured — they
    log a warning and return False instead of crashing the pipeline.
    """

    def __init__(self):
        self._blob_client:   Optional[BlobServiceClient] = None
        self._cosmos_client: Optional[object] = None
        self._alerts_container: Optional[object] = None
        self.enabled = False

        if not AZURE_AVAILABLE:
            return

        conn_str      = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        cosmos_url    = os.getenv("AZURE_COSMOS_ENDPOINT", "")
        cosmos_key    = os.getenv("AZURE_COSMOS_KEY", "")
        cosmos_db     = os.getenv("AZURE_COSMOS_DATABASE", "SurveillanceDB")
        cosmos_cont   = os.getenv("AZURE_COSMOS_CONTAINER", "Alerts")

        if not (conn_str and cosmos_url and cosmos_key):
            logger.info("Azure credentials not set — running in local-only mode.")
            return

        try:
            self._blob_client = BlobServiceClient.from_connection_string(conn_str)
            self._ensure_containers()

            cosmos          = CosmosClient(cosmos_url, cosmos_key)
            db              = cosmos.get_database_client(cosmos_db)
            self._alerts_container = db.get_container_client(cosmos_cont)

            self.enabled = True
            logger.info("✅ Azure client ready — Blob Storage + Cosmos DB connected.")
        except Exception as exc:
            logger.warning("Azure init failed (continuing locally): %s", exc)

    # ── Blob Storage ──────────────────────────────────────────────────────

    def _ensure_containers(self):
        """Create blob containers if they don't exist."""
        for name in [
            os.getenv("AZURE_STORAGE_CONTAINER_ALERTS",  "alert-frames"),
            os.getenv("AZURE_STORAGE_CONTAINER_RESULTS", "session-results"),
        ]:
            try:
                self._blob_client.create_container(name)
            except Exception:
                pass  # Already exists

    def upload_frame(self, image_path: str, alert_id: str) -> bool:
        """Upload an annotated alert frame to Blob Storage."""
        if not self.enabled:
            return False
        try:
            container = os.getenv("AZURE_STORAGE_CONTAINER_ALERTS", "alert-frames")
            blob_name = f"{datetime.now().strftime('%Y/%m/%d')}/{alert_id}.jpg"

            with open(image_path, "rb") as data:
                self._blob_client.get_blob_client(
                    container=container, blob=blob_name
                ).upload_blob(data, overwrite=True)

            logger.debug("Uploaded frame → blob://%s/%s", container, blob_name)
            return True
        except Exception as exc:
            logger.warning("Blob upload failed for %s: %s", alert_id, exc)
            return False

    def upload_session_results(self, results: dict) -> bool:
        """Upload full session JSON to Blob Storage."""
        if not self.enabled:
            return False
        try:
            container = os.getenv("AZURE_STORAGE_CONTAINER_RESULTS", "session-results")
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"session_{ts}.json"

            self._blob_client.get_blob_client(
                container=container, blob=blob_name
            ).upload_blob(
                json.dumps(results, indent=2, default=str).encode(),
                overwrite=True
            )
            logger.info("Session results uploaded → blob://%s/%s", container, blob_name)
            return True
        except Exception as exc:
            logger.warning("Session upload failed: %s", exc)
            return False

    # ── Cosmos DB ─────────────────────────────────────────────────────────

    def save_alert(self, alert_dict: dict) -> bool:
        """
        Write one alert to Cosmos DB.
        Called by alert_manager.py for every HIGH/CRITICAL alert.
        """
        if not self.enabled or self._alerts_container is None:
            return False
        try:
            doc = {
                "id":        alert_dict.get("alert_id", f"alert_{datetime.now().timestamp()}"),
                "priority":  alert_dict.get("priority", "LOW"),   # partition key
                "frame_id":  alert_dict.get("frame_id", 0),
                "timestamp": alert_dict.get("timestamp", datetime.now().isoformat()),
                "anomaly_score":   alert_dict.get("anomaly_score", 0),
                "reasons":         alert_dict.get("reasons", []),
                "detection_count": alert_dict.get("detection_count", 0),
                "motion_score":    alert_dict.get("motion_score"),
                "notified":        alert_dict.get("notified", False),
                "source":          "border_surveillance_ai",
                "version":         "v9",
            }
            self._alerts_container.upsert_item(doc)
            logger.debug("Alert saved to Cosmos DB: %s", doc["id"])
            return True
        except Exception as exc:
            logger.warning("Cosmos DB write failed: %s", exc)
            return False

    def query_recent_alerts(self, limit: int = 50) -> list:
        """Fetch recent alerts from Cosmos DB for dashboard."""
        if not self.enabled or self._alerts_container is None:
            return []
        try:
            query = (
                f"SELECT TOP {limit} * FROM c "
                f"ORDER BY c._ts DESC"
            )
            return list(self._alerts_container.query_items(
                query=query, enable_cross_partition_query=True
            ))
        except Exception as exc:
            logger.warning("Cosmos DB query failed: %s", exc)
            return []


# Singleton — import and reuse across modules
azure = AzureClient()
