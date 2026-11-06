"""Generate synthetic demo CSVs. No real personal or financial information."""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

rng = random.Random(42)
destination = Path(__file__).resolve().parents[1] / ".runtime" / "demo"
destination.mkdir(parents=True, exist_ok=True)
rows = []
for index in range(2500):
    amount = round(rng.expovariate(1 / 65), 2)
    international = rng.choice([0, 0, 0, 0, 1])
    channel = rng.choice(["web", "store", "mobile"])
    risk = 0.01 + 0.20 * international + 0.22 * (amount > 180) + 0.025 * (channel == "web")
    rows.append([
        f"T{index:05}", f"A{rng.randrange(400):03}",
        (datetime(2026, 1, 1) + timedelta(minutes=index)).isoformat(),
        "" if rng.random() < 0.04 else amount,
        "" if rng.random() < 0.03 else channel,
        international, int(rng.random() < risk),
    ])
header = ["transaction_id", "account_id", "timestamp", "amount", "channel", "international", "fraud"]
for name, data, labeled in [("training.csv", rows[:2000], True),
                            ("prediction.csv", rows[2000:], False),
                            ("prediction_labeled.csv", rows[2000:], True)]:
    with (destination / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header if labeled else header[:-1])
        writer.writerows(data if labeled else [row[:-1] for row in data])
print(f"Synthetic CSVs written to {destination}")
