# amex-default-prediction

## quickstart

Install dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure pre-commit.

```bash
pre-commit install
```

## work log

```bash
luigi --local-scheduler --module workflows.transform_parquet
```

### resource

- https://www.markhneedham.com/blog/2017/03/25/luigi-externalprogramtask-example-converting-json-csv/
- https://luigi.readthedocs.io/en/stable/example_top_artists.html
