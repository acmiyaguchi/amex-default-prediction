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
python -m workflow.model logistic  # and others
```

I spent 6 hours or so trying to use sparktorch on my local machine. Much trial
and error has lead to a few developments...

I configured pytorch lightning and petastorm using the v2 dataset. It seems to
run at around 60-70 it/s, which seems to be fairly slow. I'll try loading the
data directly from parquet, otherwise just suffer with the slow epochs.

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
- https://stackoverflow.com/questions/41399399/serialize-a-custom-transformer-using-python-to-be-used-within-a-pyspark-ml-pipel
- https://stackoverflow.com/questions/37270446/how-to-create-a-custom-estimator-in-pyspark
- https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
- https://github.com/pytorch/pytorch/issues/25767
- https://github.com/Lightning-AI/lightning/issues/1586
