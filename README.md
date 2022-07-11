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
- https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/util.html#MLWriter
- https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.util.MLWritable.html#pyspark.ml.util.MLWritable
- https://stackoverflow.com/questions/40284214/is-there-any-means-to-serialize-custom-transformer-in-spark-ml-pipeline
- https://spark.apache.org/docs/2.4.0/sql-pyspark-pandas-with-arrow.html
- https://stackoverflow.com/questions/49623620/what-type-should-the-dense-vector-be-when-using-udf-function-in-pyspark
- https://csyhuang.github.io/2020/08/01/custom-transformer/
