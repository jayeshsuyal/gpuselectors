"""Configuration constants for InferenceAtlas recommendation engine.

This module centralizes all timing constants, scaling parameters, and traffic
pattern definitions used across the cost and utilization models.
"""

from __future__ import annotations

# Time constants (30-day month model)
HOURS_PER_MONTH = 720
DAYS_PER_MONTH = 30
SECONDS_PER_DAY = 86_400

# Multi-GPU scaling parameters
U_TARGET = 0.75  # Target utilization ceiling (75% headroom rule)
MAX_GPUS = 8  # Maximum GPU count for scaling recommendations

# Traffic pattern profiles
# Each pattern defines:
# - active_ratio: fraction of time with active traffic
# - efficiency: GPU batching/scheduling efficiency factor
# - burst_factor: peak-to-average traffic multiplier
# - batch_mult: batching throughput gain under load
TRAFFIC_PATTERNS = {
    "steady": {
        "active_ratio": 1.0,
        "efficiency": 0.85,  # High batching efficiency with steady load
        "burst_factor": 1.0,
        "batch_mult": 1.25,
    },
    "business_hours": {
        "active_ratio": 0.238,  # 40 hours / 168 hours per week
        "efficiency": 0.80,  # Moderate batching during work hours
        "burst_factor": 1.0,
        "batch_mult": 1.10,
    },
    "bursty": {
        "active_ratio": 0.40,
        "efficiency": 0.70,  # Lower efficiency, irregular load
        "burst_factor": 3.0,  # 3x peak traffic during bursts
        "batch_mult": 1.35,  # Higher batching gains under load
    },
}

# Display labels used in UI flows.
TRAFFIC_PATTERN_LABELS = {
    "steady": "Steady",
    "business_hours": "Business Hours",
    "bursty": "Bursty",
}

# Planner peak-to-average defaults (kept separate from legacy burst_factor).
TRAFFIC_PATTERN_PEAK_TO_AVG_DEFAULT = {
    "steady": 1.5,
    "business_hours": 2.5,
    "bursty": 3.5,
}
