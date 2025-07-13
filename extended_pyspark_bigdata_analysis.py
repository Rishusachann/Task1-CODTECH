
# 📌 Step 1: Initialize Spark Session
from pyspark.sql import SparkSession

spark = SparkSession.builder     .appName("Extended Big Data Analysis - Boston Housing")     .getOrCreate()

# Step 2: Load the Dataset
df = spark.read.csv("boston.csv", header=True, inferSchema=True)

# Step 3: View Schema and Preview Data
df.printSchema()
df.show(5)

# Step 4: Summary Statistics
df.describe().show()

# Step 5: Check for Missing Values
from pyspark.sql.functions import col, sum
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

# Step 6: Count Distinct Values per Column
for column in df.columns:
    df.select(column).distinct().count()

# Step 7: Compute Correlation Matrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

assembler = VectorAssembler(inputCols=[c for c in df.columns if c != 'MEDV'], outputCol="features")
vec_df = assembler.transform(df).select("features")
correlation = Correlation.corr(vec_df, "features").head()
print("Correlation Matrix:\n", correlation[0])

# Step 8: Compute Skewness and Kurtosis
from pyspark.sql.functions import skewness, kurtosis
df.select([skewness(c).alias(f"{c}_skew") for c in df.columns]).show()
df.select([kurtosis(c).alias(f"{c}_kurt") for c in df.columns]).show()

# Step 9: Binning Feature 'RM' (rooms per dwelling)
from pyspark.sql.functions import when
df = df.withColumn("RM_bin", when(df["RM"] < 5, "Low")
                             .when(df["RM"] < 6.5, "Medium")
                             .otherwise("High"))

# Step 10: Aggregation by Binned Feature
df.groupBy("RM_bin").avg("MEDV").show()

# Step 11: Z-score Outlier Detection
from pyspark.sql.functions import mean, stddev, abs
stats = df.select(mean("MEDV").alias("mean"), stddev("MEDV").alias("std")).collect()
mean_val, std_val = stats[0]["mean"], stats[0]["std"]
df = df.withColumn("z_score", (df["MEDV"] - mean_val) / std_val)

# Step 12: Filter Outliers Based on Z-score
df = df.filter(abs(df["z_score"]) < 3)

# Step 13: Feature Engineering - Interaction Term (RM * LSTAT)
df = df.withColumn("RM_LSTAT", df["RM"] * df["LSTAT"])

# Step 14: Create Ratio Feature (TAX to RM)
df = df.withColumn("TAX_RM_ratio", df["TAX"] / df["RM"])

# Step 15: Log Transformation (for skewed features)
from pyspark.sql.functions import log1p
df = df.withColumn("log_CRIM", log1p("CRIM"))

# Step 16: Normalization
from pyspark.ml.feature import MinMaxScaler

assembler_norm = VectorAssembler(inputCols=[c for c in df.columns if c not in ['MEDV', 'RM_bin', 'z_score']], outputCol="raw_features")
assembled_norm = assembler_norm.transform(df)

scaler = MinMaxScaler(inputCol="raw_features", outputCol="normalized_features")
scaler_model = scaler.fit(assembled_norm)
normalized_data = scaler_model.transform(assembled_norm)

# Step 17: PCA Reduction
from pyspark.ml.feature import PCA
pca = PCA(k=5, inputCol="normalized_features", outputCol="pca_features")
pca_model = pca.fit(normalized_data)
pca_result = pca_model.transform(normalized_data)

# Step 18: Train-Test Split
train_data, test_data = pca_result.select("pca_features", "MEDV").randomSplit([0.8, 0.2], seed=123)

# Step 19: Linear Regression Model
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="pca_features", labelCol="MEDV")
lr_model = lr.fit(train_data)

# Step 20: Model Evaluation
eval_result = lr_model.evaluate(test_data)
print("Linear Regression RMSE:", eval_result.rootMeanSquaredError)
print("R²:", eval_result.r2)

# Step 21: Decision Tree Regressor
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol="raw_features", labelCol="MEDV")
dt_model = dt.fit(assembled_norm.select("raw_features", "MEDV"))

# Step 22: Random Forest Regressor
from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(featuresCol="raw_features", labelCol="MEDV", numTrees=50)
rf_model = rf.fit(assembled_norm.select("raw_features", "MEDV"))

# Step 23: Feature Importances from Random Forest
print("Feature Importances:", rf_model.featureImportances)

# Step 24: SQL Query Example
assembled_norm.createOrReplaceTempView("boston_extended")
spark.sql("SELECT RM_bin, COUNT(*) as count, AVG(MEDV) as avg_medv FROM boston_extended GROUP BY RM_bin").show()

# Step 25: Bucketing 'AGE'
df = df.withColumn("AGE_group", when(df["AGE"] < 35, "Young")
                                .when(df["AGE"] < 70, "Middle")
                                .otherwise("Old"))

# Step 26: Aggregation by Age Group
df.groupBy("AGE_group").avg("MEDV").show()

# Step 27: Heatmap-like Correlation for Visual Insight (Pseudo)
# Not actual heatmap but printing highest correlations with MEDV
for col_name in df.columns:
    if col_name != "MEDV" and df.schema[col_name].dataType != "StringType":
        corr = df.stat.corr("MEDV", col_name)
        print(f"Correlation(MEDV, {col_name}): {corr}")

# Step 28: Add Unique Row ID
from pyspark.sql.functions import monotonically_increasing_id
df = df.withColumn("row_id", monotonically_increasing_id())

# Step 29: Cache and Persist Large Data
df.cache()
df.count()

# Step 30: Save Cleaned Data for Future Use
df.write.mode("overwrite").option("header", True).csv("cleaned_boston_data")

# Step 31: Stop Spark Session
spark.stop()
