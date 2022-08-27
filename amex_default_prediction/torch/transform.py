from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.sql import functions as F


def transform_vector_to_array(df, partitions=32):
    """Cast the features and labels fields from the v2 transformed dataset to
    align with the expectations of torch."""
    return (
        df.withColumn("features", vector_to_array("features").cast("array<float>"))
        .withColumn("label", F.col("label").cast("long"))
        .repartition(partitions)
    )


def pad_src(field, pad, length):
    return F.concat(
        F.array_repeat(pad, F.lit(length) - F.size(field)),
        F.col(field),
    )


def pad_tgt(field, pad, length):
    return F.concat(
        F.col(field),
        F.array_repeat(pad, F.lit(length) - F.size(field)),
    )


def features_age_list(df, age_months=False):
    w = Window.partitionBy("customer_ID").orderBy("age_days")
    return (
        df.withColumn("features", vector_to_array("features").cast("array<float>"))
        .select(
            "customer_ID",
            F.collect_list("features").over(w).alias("features_list"),
            # age encoded into a sequence position
            F.collect_list(
                (
                    (F.col("age_days") / 28 if age_months else F.col("age_days")) + 1
                ).astype("long")
            )
            .over(w)
            .alias("age_days_list"),
        )
        .groupBy("customer_ID")
        .agg(
            F.max("features_list").alias("features_list"),
            F.max("age_days_list").alias("age_days_list"),
        )
        .withColumn("n", F.size("features_list"))
        .withColumn("dim", F.size(F.col("features_list")[0]))
    )


def transform_into_transformer_pairs(df, length=4, age_months=False):
    """Convert the training/test dataset for use in a transformer."""

    def slice_src(field, length):
        return F.when(
            F.col("n") <= length,
            F.slice(F.col(field), 1, F.col("n") - 1),
        ).otherwise(
            F.when(
                F.col("n") <= 2 * length,
                F.slice(F.col(field), 1, length),
            ).otherwise(F.slice(F.col(field), -(2 * length) + 1, length))
        )

    def slice_tgt(field, length):
        return F.when(F.col("n") <= length, F.slice(F.col(field), -1, 1)).otherwise(
            F.when(
                F.col("n") <= 2 * length, F.slice(F.col(field), length, length)
            ).otherwise(F.slice(F.col(field), -length, length))
        )

    return (
        features_age_list(df, age_months=age_months)
        .where("n > 1")
        # this is not pleasant to read, but at least it doesn't require a UDF...
        .withColumn("src", slice_src("features_list", length))
        .withColumn("tgt", slice_tgt("features_list", length))
        .withColumn("src_pos", slice_src("age_days_list", length))
        .withColumn("tgt_pos", slice_tgt("age_days_list", length))
        # create padding mask before we actually pad src/tgt
        .withColumn("k_src", F.size("src"))
        .withColumn("k_tgt", F.size("tgt"))
        # pad src and tgt with arrays filled with zeroes
        .withColumn(
            "src", pad_src("src", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn(
            "tgt", pad_tgt("tgt", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn("src_pos", pad_src("src_pos", F.lit(0), length))
        .withColumn("tgt_pos", pad_tgt("tgt_pos", F.lit(0), length))
        .withColumn(
            "src_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_src")),
                F.array_repeat(F.lit(0), F.col("k_src")),
            ),
        )
        .withColumn(
            "tgt_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(0), F.col("k_tgt")),
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_tgt")),
            ),
        )
        # now lets flatten the src and tgt rows
        .withColumn("src", F.flatten("src"))
        .withColumn("tgt", F.flatten("tgt"))
        .select(
            "customer_ID",
            "src",
            "tgt",
            "src_key_padding_mask",
            "tgt_key_padding_mask",
            "src_pos",
            "tgt_pos",
        )
    )


def transform_into_transformer_predict_pairs(df, length=4, age_months=False):
    def slice_src(field, length):
        return F.when(F.col("n") <= length, F.col(field)).otherwise(
            F.slice(F.col(field), -length, length)
        )

    return (
        features_age_list(df, age_months=age_months)
        .where("n >= 1")
        .withColumn("src", slice_src("features_list", length))
        .withColumn("src_pos", slice_src("age_days_list", length))
        # measure the length before creating a mask and padding
        .withColumn("k_src", F.size("src"))
        .withColumn(
            "src", pad_src("src", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn("src_pos", pad_src("src_pos", F.lit(0), length))
        .withColumn(
            "src_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_src")),
                F.array_repeat(F.lit(0), F.col("k_src")),
            ),
        )
        .withColumn("src_array", F.col("src"))
        .withColumn("src", F.flatten("src"))
        .select(
            "customer_ID",
            "src",
            "src_key_padding_mask",
            "src_pos",
            "src_array",
        )
    )


def transform_into_transformer_reverse_pairs(df, length=4, age_months=False):
    # drop the last item of the src column, and make the tgt the reverse of the src
    def drop_last(field, length=1):
        field = F.col(field) if isinstance(field, str) else field
        return F.slice(field, 1, F.size(field) - length)

    pairs = transform_into_transformer_predict_pairs(df, length + 1, age_months)
    return pairs.where("n > 1").select(
        "customer_ID",
        F.flatten(drop_last("src_array")).alias("src"),
        drop_last("src_key_padding_mask").alias("src_key_padding_mask"),
        drop_last("src_pos").alias("src_pos"),
        F.flatten(drop_last(F.reverse(F.col("src_array")))).alias("tgt"),
        drop_last(F.reverse(F.col("src_key_padding_mask"))).alias(
            "tgt_key_padding_mask"
        ),
        drop_last(F.reverse(F.col("src_pos"))).alias("tgt_pos"),
    )
