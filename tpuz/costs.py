"""
TPU pricing and cost tracking.
"""

import time

# Hourly rates (USD) — approximate, spot/preemptible pricing
# Source: cloud.google.com/tpu/pricing (as of 2026)
SPOT_RATES = {
    "v4-8": 2.06, "v4-16": 4.12, "v4-32": 8.24, "v4-64": 16.48,
    "v4-128": 32.96, "v4-256": 65.92, "v4-512": 131.84,
    "v5litepod-1": 1.20, "v5litepod-4": 4.80, "v5litepod-8": 9.60,
    "v5litepod-16": 19.20, "v5litepod-64": 76.80, "v5litepod-256": 307.20,
    "v5p-8": 4.20, "v5p-16": 8.40, "v5p-32": 16.80,
    "v5p-64": 33.60, "v5p-128": 67.20,
    "v6e-1": 1.20, "v6e-4": 4.80, "v6e-8": 9.60,
    "v6e-16": 19.20, "v6e-64": 76.80, "v6e-256": 307.20,
}

ON_DEMAND_RATES = {k: v * 3.0 for k, v in SPOT_RATES.items()}  # ~3x spot


def hourly_rate(accelerator, preemptible=True):
    """Get hourly rate for an accelerator type."""
    rates = SPOT_RATES if preemptible else ON_DEMAND_RATES
    return rates.get(accelerator, 0.0)


class CostTracker:
    """Track cumulative cost of a TPU VM session."""

    def __init__(self, accelerator, preemptible=True):
        self.accelerator = accelerator
        self.preemptible = preemptible
        self.rate = hourly_rate(accelerator, preemptible)
        self._start = None
        self._total_seconds = 0.0
        self._running = False

    def start(self):
        self._start = time.time()
        self._running = True

    def stop(self):
        if self._running and self._start:
            self._total_seconds += time.time() - self._start
            self._running = False

    @property
    def elapsed_hours(self):
        total = self._total_seconds
        if self._running and self._start:
            total += time.time() - self._start
        return total / 3600

    @property
    def cost(self):
        return self.elapsed_hours * self.rate

    def summary(self):
        kind = "spot" if self.preemptible else "on-demand"
        return (f"${self.cost:.2f} ({self.elapsed_hours:.2f}h × "
                f"${self.rate:.2f}/hr {self.accelerator} {kind})")

    def __repr__(self):
        return f"CostTracker({self.summary()})"
