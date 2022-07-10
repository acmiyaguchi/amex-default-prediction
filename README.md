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
python -m workflow.transform
```

### resource

- https://www.markhneedham.com/blog/2017/03/25/luigi-externalprogramtask-example-converting-json-csv/
- https://luigi.readthedocs.io/en/stable/example_top_artists.html
- https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html#pyspark.ml.feature.VectorAssembler
- https://spark.apache.org/docs/latest/monitoring.html#viewing-after-the-fact
- https://spark.apache.org/docs/latest/ml-pipeline.html#example-pipeline
