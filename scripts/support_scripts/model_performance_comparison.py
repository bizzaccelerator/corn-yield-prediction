import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from evidently.ui.workspace import RemoteWorkspace

# ======= CONFIG =======
EVIDENTLY_SERVICE_URL = "https://evidently-ui-453290981886.us-central1.run.app"
GCS_BUCKET_NAME = "corn-yield-prediction-kenia-evidently-reports"
PROJECT_NAME = "Corn Yield ML Monitoring"

# Alert configuration
ALERT_CONFIG = {
    "rmse_degradation_threshold": float(
        os.getenv("RMSE_DEGRADATION_THRESHOLD", "0.05")
    ),  # 5% increase triggers alert
    "r2_degradation_threshold": float(
        os.getenv("R2_DEGRADATION_THRESHOLD", "0.05")
    ),  # 5% decrease triggers alert
    "mae_degradation_threshold": float(
        os.getenv("MAE_DEGRADATION_THRESHOLD", "0.05")
    ),  # 5% increase triggers alert
    "drift_alert_enabled": True,
    "performance_alert_enabled": True,
}

# Email alert configuration
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "recipient_emails": os.getenv("RECIPIENT_EMAILS", "").split(","),
}
# ======================


def to_float(value: Any) -> Optional[float]:
    """Robust conversion from weird string numeric formats to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return None

    # Try to find a numeric token (handles commas/dots, thousands separators)
    m = re.search(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|[-+]?\d*\.?\d+", s)
    if not m:
        return None
    num = m.group(0)

    # Normalize separators:
    # If both . and , exist, assume last is decimal separator:
    if num.count(".") > 0 and num.count(",") > 0:
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    else:
        num = num.replace(",", ".")

    try:
        return float(num)
    except Exception:
        return None


def safe_parse_iso(ts: Optional[str]) -> datetime:
    """Parse ISO timestamps robustly; fallback to minimal date when unparsable."""
    if not ts:
        return datetime.min
    try:
        # Python 3.7+ supports fromisoformat (handles microseconds)
        return datetime.fromisoformat(ts)
    except Exception:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
    return datetime.min


def get_or_create_project(ws: RemoteWorkspace, project_name: str):
    """Return (project_obj or None)."""
    try:
        projects = ws.list_projects()
        for proj in projects:
            if getattr(proj, "name", "") == project_name:
                return proj
        print(f"[warn] Project not found: '{project_name}'")
        return None
    except Exception as e:
        print(f"[error] Error listing projects: {e}")
        return None


def get_report_list_from_evidently(project_id: str) -> List[str]:
    """
    Query the Evidently UI REST endpoint for reports list.
    Returns list of report IDs (strings).
    """
    api_url = f"{EVIDENTLY_SERVICE_URL}/api/projects/{project_id}/reports"
    try:
        resp = requests.get(api_url, timeout=15)
        if resp.status_code != 200:
            print(f"[error] /reports returned HTTP {resp.status_code}: {resp.text}")
            return []
        data = resp.json()
        # handle possible nested { items: [...] } structure
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            print("[warn] Unexpected reports list format")
            return []
        ids = [it.get("id") for it in items if it.get("id")]
        print(f"[info] Found {len(ids)} report IDs in Evidently service")
        return ids
    except Exception as e:
        print(f"[error] Error fetching reports list: {e}")
        return []


def fetch_report_from_gcs(project_id: str, report_id: str) -> Optional[Dict]:
    """Download JSON snapshot from public GCS URL (or authenticated if bucket is private)."""
    gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{project_id}/snapshots/{report_id}.json"
    try:
        resp = requests.get(gcs_url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        print(f"[warn] GCS fetch HTTP {resp.status_code} for {report_id}")
        return None
    except Exception as e:
        print(f"[error] Error fetching from GCS: {e}")
        return None


def extract_performance_metrics(report_data: Dict) -> Dict[str, Any]:
    """
    Extracts performance metrics from an Evidently report JSON.
    Produces keys:
      - rmse_current, rmse_reference, rmse (preferred current)
      - r2_current, r2_reference, r2
      - mae_current, mae_reference, mae
      - drift_detected (bool)
      - report_id, timestamp, name
    """
    results = {
        "report_id": report_data.get("id"),
        "timestamp": report_data.get("timestamp"),
        "name": report_data.get("name"),
    }
    found = {}  # temporary store per metric

    # helper to register extracted numeric into found structure
    def register(metric_key, which, val, debug_ctx=""):
        if val is None:
            return
        found.setdefault(metric_key, {})
        found[metric_key][which] = val
        print(f"    debug: registered {metric_key}.{which} = {val} {debug_ctx}")

    # Primary source: metric_results (structured)
    metric_results = (
        report_data.get("metric_results")
        or report_data.get("report", {}).get("metric_results")
        or {}
    )
    if isinstance(metric_results, dict) and metric_results:
        for m_id, m_obj in metric_results.items():
            display = str(m_obj.get("display_name") or "").lower()
            key = None
            if "rmse" in display:
                key = "rmse"
            elif "r2" in display or "r2 score" in display or "r-squared" in display:
                key = "r2"
            elif "mean absolute error" in display or "mae" in display:
                key = "mae"
            else:
                # Not a performance metric we're interested in
                continue

            # Inspect widget counters
            widgets = m_obj.get("widget") or []
            for widget in widgets:
                title = str(widget.get("title") or "").lower()
                params = widget.get("params") or {}
                counters = params.get("counters") or []
                for counter in counters:
                    raw_val = counter.get("value") or counter.get(
                        "label"
                    )  # sometimes label holds text
                    numeric = to_float(raw_val)

                    # Prefer explicit "current" / "reference" in title
                    if "current" in title:
                        register(
                            key,
                            "current",
                            numeric,
                            f"(widget title='{widget.get('title')}')",
                        )
                    elif "reference" in title:
                        register(
                            key,
                            "reference",
                            numeric,
                            f"(widget title='{widget.get('title')}')",
                        )
                    else:
                        # label-driven fallbacks
                        label = str(counter.get("label") or "").lower()
                        if "mean" in label or label == "" or "score" in label:
                            # a generic numeric: keep as 'value' fallback
                            register(
                                key,
                                "value",
                                numeric,
                                f"(label='{counter.get('label')}')",
                            )
                        else:
                            # keep as fallback under the label name
                            register(
                                key, label or "value", numeric, f"(label fallback)"
                            )

            # If metric_result has direct 'value' (scalar), keep as fallback
            if "value" in m_obj and m_obj.get("value") is not None:
                register(
                    key, "value", to_float(m_obj.get("value")), "(metric_result.value)"
                )

            # Some metrics use 'mean' or 'std' keys
            if (
                isinstance(m_obj.get("mean"), dict)
                and m_obj["mean"].get("value") is not None
            ):
                register(
                    key,
                    "value",
                    to_float(m_obj["mean"].get("value")),
                    "(metric_result.mean.value)",
                )
            if (
                isinstance(m_obj.get("count"), dict)
                and m_obj["count"].get("value") is not None
            ):
                # not performance but we keep for completeness
                register(key, "count", to_float(m_obj["count"].get("value")), "(count)")

    # Build final chosen metrics (prefer current -> value -> reference)
    for metric_key in ("rmse", "r2", "mae"):
        metric_info = found.get(metric_key, {})
        chosen = None
        if metric_info.get("current") is not None:
            chosen = metric_info.get("current")
        elif metric_info.get("value") is not None:
            chosen = metric_info.get("value")
        elif metric_info.get("reference") is not None:
            chosen = metric_info.get("reference")
        else:
            chosen = None

        results[f"{metric_key}_current"] = metric_info.get("current")
        results[f"{metric_key}_reference"] = metric_info.get("reference")
        results[metric_key] = chosen

    drift_detected = False
    try:
        # metric_results text search
        dump = json.dumps(metric_results)
        if "drift" in dump.lower() and (
            "detected" in dump.lower() or "drift" in dump.lower()
        ):
            if "detected" in dump.lower():
                drift_detected = True
            else:
                # try to detect drift_score > 0.0
                m = re.search(r'"drift_score"\s*:\s*([0-9\.eE+-]+)', dump)
                if m and float(m.group(1)) > 0.0:
                    drift_detected = True
    except Exception:
        drift_detected = False

    results["drift_detected"] = drift_detected

    # Debug summary
    print(
        f"  Extracted for report {
            results.get('report_id')}: rmse={
            results.get('rmse')}, r2={
                results.get('r2')}, mae={
                    results.get('mae')}, drift={
                        results.get('drift_detected')}")
    return results


def calculate_performance_change(
    baseline: float, current: float, metric_name: str
) -> Tuple[Optional[float], Optional[float], str]:
    """Calculate absolute and percentage change. Returns (abs_change, pct_change, direction_str)."""
    if baseline is None or current is None:
        return None, None, "unknown"

    if baseline == 0:
        pct_change = float("inf") if current != 0 else 0.0
    else:
        pct_change = ((current - baseline) / abs(baseline)) * 100.0

    absolute_change = current - baseline
    if metric_name == "r2":
        direction = (
            "improvement"
            if absolute_change > 0
            else "degradation"
            if absolute_change < 0
            else "no_change"
        )
    else:  # RMSE/MAE lower is better
        direction = (
            "improvement"
            if absolute_change < 0
            else "degradation"
            if absolute_change > 0
            else "no_change"
        )

    return absolute_change, pct_change, direction


def send_email_alert(message: str, subject: str) -> bool:
    """Send email alert using SMTP."""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.utils import formataddr

        # Validate configuration
        if not all(
            [
                EMAIL_CONFIG["smtp_server"],
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["sender_password"],
            ]
        ):
            print(
                "[warn] Email configuration incomplete (missing SMTP server, sender email, or password)"
            )
            return False

        # Filter out empty recipient emails
        recipients = [
            email.strip() for email in EMAIL_CONFIG["recipient_emails"] if email.strip()
        ]
        if not recipients:
            print("[warn] No valid recipient emails configured")
            return False

        # Create email message
        msg = MIMEMultipart("alternative")
        msg["From"] = formataddr(("ML Monitoring System", EMAIL_CONFIG["sender_email"]))
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        # Create HTML and plain text versions
        html_body = f"""
        <html>
          <head></head>
          <body>
            <h2 style="color: #d32f2f;">🚨 ML Model Alert - Corn Yield Prediction</h2>
            <div style="background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336; margin: 10px 0;">
              <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">{message}</pre>
            </div>
            <hr>
            <p><strong>Dashboard:</strong> <a href="{EVIDENTLY_SERVICE_URL}
            " target="_blank">{
            EVIDENTLY_SERVICE_URL}</a></p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            <p style="color: #666; font-size: 12px;">This is an automated alert from the ML monitoring system.</p>
          </body>
        </html>
        """

        plain_body = f"""
        ML Model Alert - Corn Yield Prediction
        =====================================

        {message}

        Dashboard: {EVIDENTLY_SERVICE_URL}

        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

        ---
        This is an automated alert from the ML monitoring system.
        """

        # Attach both versions
        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Send email
        print(f"[info] Sending email alert to: {', '.join(recipients)}")

        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()  # Enable TLS encryption
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
        server.send_message(msg)
        server.quit()

        print(f"[info] Email alert sent successfully")
        return True

    except Exception as e:
        print(f"[error] Failed to send email alert: {e}")
        return False


def check_and_send_alerts(baseline: Dict, current: Dict, comparison_results: Dict):
    """Check for alerting conditions and send email notifications."""
    alerts_sent = []
    alert_messages = []

    # Check for drift alerts
    if (
        ALERT_CONFIG["drift_alert_enabled"]
        and current.get("drift_detected", False)
        and not baseline.get("drift_detected", False)
    ):
        message = (
            f"🔍 DATA DRIFT DETECTED\n\n"
            f"New drift has been detected in the current model report.\n\n"
            f"Baseline report: {baseline.get('report_id')} (no drift)\n"
            f"Current report: {current.get('report_id')} (drift detected)\n\n"
            f"Please investigate the data quality and model performance immediately."
        )
        alert_messages.append(message)

    # Check for performance degradation alerts
    if ALERT_CONFIG["performance_alert_enabled"]:
        performance_alerts = []

        for metric in ["rmse", "r2", "mae"]:
            threshold_key = f"{metric}_degradation_threshold"
            if threshold_key not in ALERT_CONFIG:
                continue

            baseline_val = baseline.get(metric)
            current_val = current.get(metric)

            if baseline_val is None or current_val is None:
                continue

            _, pct_change, direction = calculate_performance_change(
                baseline_val, current_val, metric
            )

            if pct_change is None:
                continue

            threshold = ALERT_CONFIG[threshold_key] * 100  # Convert to percentage

            # Check if degradation exceeds threshold
            is_degraded = False
            if metric == "r2":  # Higher is better for R2
                is_degraded = direction == "degradation" and abs(pct_change) > threshold
            else:  # Lower is better for RMSE/MAE
                is_degraded = direction == "degradation" and abs(pct_change) > threshold

            if is_degraded:
                performance_alerts.append(
                    f"{metric.upper()}: {baseline_val:.6f} → {current_val:.6f} "
                    f"({pct_change:+.2f}%, threshold: ±{threshold:.1f}%)"
                )

        if performance_alerts:
            message = (
                f"📉 PERFORMANCE DEGRADATION DETECTED\n\n"
                f"The following metrics have degraded beyond acceptable thresholds:\n\n"
                f"{'• ' + chr(10) + '• '.join(performance_alerts)}\n\n"
                f"Baseline report: {baseline.get('report_id')} ({baseline.get('timestamp')})\n"
                f"Current report: {current.get('report_id')} ({current.get('timestamp')})\n\n"
                f"Please review model performance and consider retraining."
            )
            alert_messages.append(message)

    # Send email alerts if any conditions were met
    for i, message in enumerate(alert_messages, 1):
        print(f"\n[ALERT {i}] Sending email notification...")
        print(f"Message preview: {message[:200]}{'...' if len(message) > 200 else ''}")

        # Create subject based on alert type
        if "DRIFT" in message:
            subject = "🔍 ML Model Alert: Data Drift Detected - Corn Yield Prediction"
        else:
            subject = (
                "📉 ML Model Alert: Performance Degradation - Corn Yield Prediction"
            )

        if send_email_alert(message, subject):
            alerts_sent.append("Email")
        else:
            print(f"[error] Failed to send email alert {i}")

    return alert_messages, alerts_sent


def main():
    print("=" * 60)
    print("MODEL PERFORMANCE COMPARISON FROM GCS REPORTS (with email alerting)")
    print("=" * 60)

    # Connect to Evidently
    try:
        ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    except Exception as e:
        print(f"[fatal] Cannot create RemoteWorkspace: {e}")
        return

    project = get_or_create_project(ws, PROJECT_NAME)
    if not project:
        print(
            "[fatal] Project not found. Ensure monitoring scripts created the project in Evidently UI."
        )
        return

    project_id = str(project.id)
    print(f"[info] Using project ID: {project_id}  (Evidently URL: {
        EVIDENTLY_SERVICE_URL}/projects/{project_id})")

    report_ids = get_report_list_from_evidently(project_id)
    if not report_ids:
        print("[fatal] No reports returned by Evidently service.")
        return

    all_metrics = []
    for rid in report_ids:
        print(f"\n[info] Fetching report: {rid}")
        report_json = fetch_report_from_gcs(project_id, rid)
        if not report_json:
            print(f"[warn] Could not fetch report {rid}")
            continue
        metrics = extract_performance_metrics(report_json)
        if any(
            metrics.get(k) is not None for k in ("rmse", "r2", "mae")
        ) or metrics.get("drift_detected"):
            all_metrics.append(metrics)
        else:
            print(f"[debug] No performance metrics found in report {rid}; skipping")

    if not all_metrics:
        print(
            "[fatal] No usable metrics extracted from any report. See troubleshooting tips in your environment."
        )
        return

    # Sort reports by timestamp (oldest -> newest)
    all_metrics.sort(key=lambda x: safe_parse_iso(x.get("timestamp")))
    baseline = all_metrics[0]
    current = all_metrics[-1]

    print("\n" + "=" * 40)
    print("Selected baseline (oldest) and current (newest) reports")
    print("=" * 40)
    print(
        f"Baseline report: {
            baseline.get('report_id')}  timestamp: {
            baseline.get('timestamp')}")
    print(
        f"Current  report: {
            current.get('report_id')}  timestamp: {
            current.get('timestamp')}")

    # Compare metrics
    print("\nPERFORMANCE COMPARISON:")
    comparison_results = {}
    made = False
    for m in ("rmse", "r2", "mae"):
        b = baseline.get(m)
        c = current.get(m)
        if b is None and c is None:
            print(f"  {m.upper()}: no data")
            continue
        abs_ch, pct_ch, direction = calculate_performance_change(b, c, m)
        if abs_ch is None:
            print(f"  {m.upper()}: insufficient data (baseline or current missing)")
            continue
        print(f"\n  {m.upper()}:")
        print(f"    Baseline: {b:.6f}")
        print(f"    Current : {c:.6f}")
        print(f"    Change  : {abs_ch:+.6f} ({pct_ch:+.2f}%)")
        print(f"    Status  : {direction.upper()}")

        comparison_results[m] = {
            "baseline": b,
            "current": c,
            "absolute_change": abs_ch,
            "percentage_change": pct_ch,
            "direction": direction,
        }
        made = True

    # Drift
    if (
        baseline.get("drift_detected") is not None
        or current.get("drift_detected") is not None
    ):
        bd = baseline.get("drift_detected", False)
        cd = current.get("drift_detected", False)
        print("\nDRIFT STATUS:")
        print(f"  Baseline: {bd}")
        print(f"  Current : {cd}")
        if cd and not bd:
            print("ALERT: New drift detected!")
        elif not cd and bd:
            print("Improvement: drift resolved")

    if not made:
        print(
            "\nNo comparable performance metric pairs found between baseline and current reports."
        )

    # Check for alerts and send notifications
    alert_messages, alerts_sent = check_and_send_alerts(
        baseline, current, comparison_results
    )

    # Prepare output for Kestra
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id,
        "baseline_report": {
            "id": baseline.get("report_id"),
            "timestamp": baseline.get("timestamp"),
            "metrics": {k: baseline.get(k) for k in ["rmse", "r2", "mae"]},
            "drift_detected": baseline.get("drift_detected", False),
        },
        "current_report": {
            "id": current.get("report_id"),
            "timestamp": current.get("timestamp"),
            "metrics": {k: current.get(k) for k in ["rmse", "r2", "mae"]},
            "drift_detected": current.get("drift_detected", False),
        },
        "comparison_results": comparison_results,
        "alerts": {
            "messages": alert_messages,
            "channels_notified": alerts_sent,
            "alert_count": len(alert_messages),
        },
    }

    # Save results to output file
    with open("performance_comparison.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n[info] Results saved to performance_comparison.json")
    if alert_messages:
        print(
            f"[info] {
                len(alert_messages)} alert(s) generated and sent to: {
                ', '.join(alerts_sent)}")

    print("\n" + "=" * 60)
    print(f"Evidently dashboard: {EVIDENTLY_SERVICE_URL}/projects/{project.id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
