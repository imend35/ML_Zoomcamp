
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

EXPECTED_COLUMNS = ["AT", "V", "AP", "RH", "PE"]
EXPECTED_ROWS = 9568


def main():
    print("Downloading Combined Cycle Power Plant dataset...")

    dataset = fetch_ucirepo(id=294)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    df = df[EXPECTED_COLUMNS]

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:", df.isna().sum().to_dict())
    print("Duplicate rows:", int(df.duplicated().sum()))

    assert len(df) == EXPECTED_ROWS, "Unexpected row count"
    assert df.columns.tolist() == EXPECTED_COLUMNS
    assert df.isna().sum().sum() == 0

    for col in EXPECTED_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col])

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "power_plant.csv"
    df.to_csv(output_path, index=False)

    print(f"PASS: Dataset validated and saved to {output_path}")


if __name__ == "__main__":
    main()
