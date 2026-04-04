"""
Notifications — Slack, email, or webhook on training events.
"""

import json
import urllib.request


def send_slack(webhook_url, message):
    """Send a Slack notification via webhook."""
    data = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        webhook_url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False


def send_webhook(url, payload):
    """Send a generic webhook POST."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"Webhook failed: {e}")
        return False


def notify(url, message):
    """Auto-detect notification type and send."""
    if not url:
        return
    if "hooks.slack.com" in url:
        send_slack(url, message)
    else:
        send_webhook(url, {"message": message})
