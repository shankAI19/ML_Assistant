from .common import load_csv
from .paths import WORK


def run(payload):
    frame = load_csv(WORK / "input.csv")
    columns = []
    for name in frame:
        series = frame[name]
        unique = int(series.nunique())
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": unique,
                "possible_id": unique == len(frame),
                "binary_candidate": unique == 2,
                "values": [str(v)[:100] for v in series.dropna().unique()[:10]]
                if unique <= 10
                else [],
                "value_counts": {str(k)[:100]: int(v) for k, v in series.value_counts().items()}
                if unique <= 10
                else {},
            }
        )
    return {
        "rows": len(frame),
        "columns": columns,
        "duplicates": int(frame.duplicated().sum()),
        "note": "Review target candidates and ID columns with the user. Profiling cannot establish whether a feature leaks future information.",
    }
