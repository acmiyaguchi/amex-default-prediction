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

Batch size of 3000: 8.7GB/11GB (55% cuda, 30% copy)
Batch size of 4000: 10.8GB/11GB (70% cuda, 40% copy)

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
- https://github.com/uber/petastorm/issues/570
- https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html

### transformer

The first iteration only takes into consideration a single window per customer.
I chose a sequence length of 8 based on the histogram of statements per customer.

```
+--------------+------+
|customer_count| count|
+--------------+------+
|             1|  5827|
|             2|  8174|
|             3|  7803|
|             4|  8348|
|             5|  8419|
|             6|  8833|
|             7|  9653|
|             8|  9775|
|             9| 10552|
|            10|  9638|
|            11|  9943|
|            12| 16327|
|            13|811329|
+--------------+------+
```

Here are the scores when using the the transformer model.

```
+-------------------------+-----------------------------+---------------------+
|model                    |version                      |bestScore            |
+-------------------------+-----------------------------+---------------------+
|gbt-with-transformer     |20220810021709-0.16.4-378ecd8|0.6478601422210726   |
|logistic-with-transformer|20220809080410-0.16.4-79896fa|0.6577801934802608   |
|logistic-with-transformer|20220809073645-0.16.3-1d6e022|0.6254525936757505   |
|gbt-with-transformer     |20220808094025-0.16.2-8a4b536|0.6338066825994494   |
+-------------------------+-----------------------------+---------------------+
```

The first two rows are from using all the features (128x8).
The last two rows are from using only the first feature (128x1).

Here are a few things that I want to try and see if there are improvements:

- Increase the sequence length to 16.
  This is easy and requires no code changes.
- Increase the number of encoder/decoder layers.
- Use a smaller position encoder utilizing age in months instead of age in days.
  This is a minor code change, and can be tested using multiple sequence lengths.
- Use a larger dataset.
  In addition to using the left-most shifted window, I'd like to also include a dataset that includes right shifted windows.
  This is a complicated change, but could improve results further.

#### models

- models/torch-transformer/20220725054744-0.16.2-6d73fff/lightning_logs_amex-default-prediction/0_21aakrzj/checkpoints/epoch=8-step=1656.ckpt
  - initial model with sequence length 8
- models/torch-transformer/20220810060736-0.17.0-36d978d/model.ckpt
  - sequence length 16
- models/torch-transformer/20220810184502-0.17.1-53636bc/model.ckpt
  - set layers to 8 (sequence 8) 0.63
- 20220810235446-0.17.2-3690d12
  - set layers to 3 (sequence 8) 0.65
- 20220811010740-0.17.2-4ca01fa
  - set layers to 3 but used age in months
- 20220811061857-0.17.3-dfa0520
  - use new data loading with reversed sequence (16). 0.625
- 20220811133617-0.17.4-4439338
  - reversed with cosine warmup scheduling
- 20220812021606-0.17.5-2df8fd5
  - reversed with cosine warmup scheduling and increased layers
