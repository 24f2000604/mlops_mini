import pandas as pd

from utils.clean_data import clean_data


def _frame():
    rows = [
        {
            "record_id": "R1",
            "date": "2026-01-01",
            "cows": 10,
            "buffaloes": 5,
            "fodder_price_rs": 11.0,
            "temp_c": 29.0,
            "vet_visits": 1,
            "feed_kg": 200.0,
            "breed_mix": "desi",
            "village_cluster": "Cluster A",
            "litres_paid_in_passbook": 100.0,
            "litres_per_day": 99.0,
        },
        {
            "record_id": "R2",
            "date": "2026-01-02",
            "cows": 12,
            "buffaloes": 6,
            "fodder_price_rs": 12.0,
            "temp_c": 30.0,
            "vet_visits": 0,
            "feed_kg": 180.0,
            "breed_mix": "crossbred",
            "village_cluster": "Cluster B",
            "litres_paid_in_passbook": 110.0,
            "litres_per_day": 105.0,
        },
    ]
    rows.append(dict(rows[1]))
    return pd.DataFrame(rows)


def test_drops_passbook_column():
    out = clean_data(_frame())
    assert "litres_paid_in_passbook" not in out.columns


def test_rescales_cluster_a_feed_kg():
    out = clean_data(_frame())
    cluster_a = out[out.village_cluster == "Cluster A"].iloc[0]
    assert cluster_a["feed_kg"] == 200.0 / 1e3


def test_drops_duplicate_rows():
    out = clean_data(_frame())
    assert len(out) == 2
