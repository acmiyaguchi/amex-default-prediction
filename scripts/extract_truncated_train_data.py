from pathlib import Path


def fetch_n_lines(path, n):
    with path.open() as f:
        return [next(f) for _ in range(n)]


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    amex_root = root / "data/raw/amex-default-prediction"
    assert amex_root.exists()
    test_root = root / "tests" / "data"
    test_root.mkdir(exist_ok=True, parents=True)

    # take the first 20 rows of the train file, including header
    n = 21
    for path in amex_root.glob("*"):
        lines = fetch_n_lines(path, n)
        (test_root / path.name).write_text("".join(lines))
