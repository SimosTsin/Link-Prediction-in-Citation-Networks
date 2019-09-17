import sys
import os
import csv

# Set the path for spark installation
# this is the path where you downloaded and uncompressed the Spark distribution
# Using forward slashes on windows, \\ should work too.
os.environ['SPARK_HOME'] = "/home/user/bdc/spark-2.1.0-bin-hadoop2.6/"
# Append the python dir to PYTHONPATH so that pyspark could be found
sys.path.append("/home/user/bdc/spark-2.1.0-bin-hadoop2.6/python/")
# Append the python/build to PYTHONPATH so that py4j could be found
sys.path.append("/home/user/bdc/spark-2.1.0-bin-hadoop2.6/python/lib/")

# try the import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark import SQLContext
    from pyspark.sql import SparkSession
    from pyspark.sql.window import Window
    from pyspark.sql.functions import udf
    from pyspark.sql.functions import struct
    from pyspark.sql.functions import split
    from pyspark.sql.functions import lit
    from pyspark.sql.functions import col
    from pyspark.sql.functions import row_number
    from pyspark.sql.types import IntegerType, FloatType, BinaryType, NullType
    from pyspark.sql.types import ArrayType
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.classification import GBTClassifier
    from pyspark.mllib.util import MLUtils

except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)

def findNumberCommonWordsTitle (string1, string2):
    set1 = set(string1)
    set2 = set(string2)
    return len(set1.intersection(set2))

def findNumberCommonAuthors (string1, string2):
    set1 = set(string1.split(','))
    set2 = set(string2.split(','))
    return len(set1.intersection(set2))

def findJaccardCoef(string1, string2):
    set1 = set(string1)
    set2 = set(string2)
    return float(len(set1 & set2)) / len(set1 | set2)

def sameJournal(string1, string2):
    set1 = set(string1)
    set2 = set(string2)
    if len(set1) != 0 and len(set2) != 0 and set1 == set2:
        return 1
    else:
        return 0

if __name__ == "__main__":

    # Create spark session
    spark = SparkSession.builder.master("local[2]").appName('link-prediction').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Create spark context
    sc = spark.sparkContext

    # Create a dataframe from training_set and testing_set
    trainingRDD = sc.textFile("training_set.txt").map(lambda x:x.strip().split(' '))
    trainingDF = trainingRDD.toDF(['from_node_id', 'to_node_id', 'label']).sample(False, 0.3, 10)
    predRDD = sc.textFile("testing_set.txt").map(lambda x:x.strip().split(' '))
    predDF = predRDD.toDF(['from_node_id', 'to_node_id'])
    predictDF = predDF.withColumn('label', lit(None).cast(NullType()))
    combinedDF = trainingDF.union(predictDF)

    print("Input DataFrame contains %d elements" % trainingDF.count())
    print("To predict DataFrame contains %d elements" % predictDF.count())
    print("The combined DataFrame contains %d elements" % combinedDF.count())

    # Create a dataframe for paper information (title, authors, abstract, etc)
    infoRDD = sc.textFile("node_information.csv")
    infoRDD = infoRDD.mapPartitions(lambda x: csv.reader(x))
    infoDF = infoRDD.toDF(['node_id','year','title','authors','journal','abstract'])
    infoDF.printSchema()
    infoDF.show(5)
    raw_input("Press enter ... ")


    # ****************************************************************************** #
    #  Using the following features:                                                 #
    #   1. number of overlapping words in paper titles                               #
    #   2. number of common authors in both papers                                   #
    #   3. temporal distance between the papers' publication years                   #
    #   4. jaccard simmilarity on abstract, title and authors                        #
    #   5. same journal                                                              #
    # ****************************************************************************** #

    # Perform a sequence of joins between infoDF and combinedDF to incorporate paper information.
    newDF = infoDF.join(combinedDF, infoDF.node_id == combinedDF.from_node_id)\
        .select('from_node_id', 'title', 'year', 'authors', 'abstract', 'journal', 'to_node_id', 'label')
    newDF = newDF.withColumnRenamed('title', 'title_from')\
        .withColumnRenamed('year', 'year_from')\
        .withColumnRenamed('authors', 'authors_from')\
        .withColumnRenamed('abstract', 'abstract_from')\
        .withColumnRenamed('journal', 'journal_from')
    newDF = infoDF.join(newDF, infoDF.node_id == newDF.to_node_id)\
        .select('from_node_id', 'title_from', 'year_from', 'authors_from', 'abstract_from', 'journal_from', 'to_node_id', 'title', 'year', 'authors', 'abstract', 'journal', 'label')
    newDF = newDF.withColumnRenamed('title', 'title_to')\
        .withColumnRenamed('year', 'year_to')\
        .withColumnRenamed('authors', 'authors_to')\
        .withColumnRenamed('abstract', 'abstract_to')\
        .withColumnRenamed('journal', 'journal_to')

    # Change the data type of specific columns.
    newDF = newDF.withColumn('year_from', newDF["year_from"].cast(IntegerType()))
    newDF = newDF.withColumn('year_to', newDF["year_to"].cast(IntegerType()))
    newDF = newDF.withColumn('label', newDF['label'].cast(IntegerType()))
    newDF.show(5)

    # ****************************************************************************** #
    # Remove stopwords from the title of both papers.                                #
    # ****************************************************************************** #
    newDF = newDF.withColumn('title_from_words', split("title_from", "\s+"))
    newDF = newDF.withColumn('title_to_words', split("title_to", "\s+"))
    newDF = newDF.withColumn('abstract_from_words', split("abstract_from", "\s+"))
    newDF = newDF.withColumn('abstract_to_words', split("abstract_to", "\s+"))
    newDF = newDF.withColumn('authors_from_words', split("authors_from", "\s+"))
    newDF = newDF.withColumn('authors_to_words', split("authors_to", "\s+"))
    remover = StopWordsRemover(inputCol='title_from_words', outputCol='title_from_words_f')
    newDF = remover.transform(newDF)
    remover = StopWordsRemover(inputCol='title_to_words', outputCol='title_to_words_f')
    newDF = remover.transform(newDF)
    remover = StopWordsRemover(inputCol='abstract_from_words', outputCol='abstract_from_words_f')
    newDF = remover.transform(newDF)
    remover = StopWordsRemover(inputCol='abstract_to_words', outputCol='abstract_to_words_f')
    newDF = remover.transform(newDF)
    remover = StopWordsRemover(inputCol='authors_from_words', outputCol='authors_from_words_f')
    newDF = remover.transform(newDF)
    remover = StopWordsRemover(inputCol='authors_to_words', outputCol='authors_to_words_f')
    newDF = remover.transform(newDF)
    newDF.show(5)

    # Definitions of User-Defined-Functions (UDFs)
    # Define a UDF to create a new column based on the values of other columns.
    udf_title_overlap = udf(lambda x: findNumberCommonWordsTitle(x[0], x[1]), returnType=IntegerType())
    udf_authors_overlap = udf(lambda x: findNumberCommonAuthors(x[0], x[1]), returnType=IntegerType())
    udf_jaccard_coef = udf(lambda x: findJaccardCoef(x[0], x[1]), returnType=FloatType())
    udf_same_journal = udf(lambda x: sameJournal(x[0], x[1]), returnType=IntegerType())
    udf_abs = udf(lambda x: abs(x[0] - x[1]), returnType=IntegerType())
    udf_tovector = udf(lambda x: Vectors.dense(x), returnType=VectorUDT())

    newDF = newDF.withColumn('title_overlap', udf_title_overlap(struct(newDF.title_from_words_f, newDF.title_to_words_f)))
    newDF = newDF.withColumn('author_overlap', udf_authors_overlap(struct(newDF.authors_from, newDF.authors_to)))
    newDF = newDF.withColumn('time_dist', udf_abs(struct(newDF.year_from, newDF.year_to)))
    newDF = newDF.withColumn('jaccard_coef', udf_jaccard_coef(struct(newDF.abstract_from_words_f, newDF.abstract_to_words_f)))
    newDF = newDF.withColumn('jaccard_coef_ath', udf_jaccard_coef(struct(newDF.authors_from_words_f, newDF.authors_to_words_f)))
    newDF = newDF.withColumn('jaccard_coef_tit', udf_jaccard_coef(struct(newDF.title_from_words_f, newDF.title_to_words_f)))
    newDF = newDF.withColumn('s_journal', udf_same_journal(struct(newDF.journal_from, newDF.journal_to)))
    newDF = newDF.withColumn('features', udf_tovector(struct(newDF.title_overlap, newDF.author_overlap, newDF.time_dist, newDF.jaccard_coef, newDF.s_journal, newDF.jaccard_coef_ath, newDF.jaccard_coef_tit)))
    newDF.printSchema()
    newDF.show(30)

    # Separate the newDF into the training and the testing set again and create the id column on the testing set.
    to_predictDF = newDF.filter(col("label").isNull())
    w = Window.orderBy("from_node_id")
    to_predictDF = to_predictDF.withColumn("id", row_number().over(w))
    to_trainDF = newDF.filter(col("label").isNotNull())

    # Create train & test sets as different dataframes from the training set.
    # Use a simple random split.
    (trainDF, testDF) = to_trainDF.randomSplit([0.6, 0.4])

    # ****************************************************************************** #
    # Run Logistic Regression Classification.                                        #
    # ****************************************************************************** #

    lr = LogisticRegression(family='binomial', featuresCol='features', labelCol='label', predictionCol='pred', rawPredictionCol='pred_raw', maxIter=10)
    lr_model = lr.fit(trainDF)
    lr_result = lr_model.transform(testDF)

    # Create an evaluator to measure classification performance.
    evaluator1 = BinaryClassificationEvaluator(rawPredictionCol='pred_raw', labelCol='label', metricName='areaUnderPR')
    area_under_pr = evaluator1.evaluate(lr_result)
    evaluator2 = MulticlassClassificationEvaluator(predictionCol="pred", labelCol="label", metricName="f1")
    f1_score = evaluator2.evaluate(lr_result)
    evaluator3 = MulticlassClassificationEvaluator(predictionCol="pred", labelCol="label", metricName="accuracy")
    accuracy = evaluator3.evaluate(lr_result)

    print("")
    print("########################################################################")
    print("LOGISTIC REGRESSION RESULTS")
    print("Area under PR curve: " + str(area_under_pr))
    print("F1 score = %g" % f1_score)
    print("Accuracy = %g" % accuracy)
    print("########################################################################")

    # Display the label and the prediction for the first 10 pairs.
    lr_result.select('label', 'pred').show(10)

    # ****************************************************************************** #
    # Run Random Forest Classification.                                              #
    # ****************************************************************************** #

    rf = RandomForestClassifier(featuresCol='features', labelCol='label', predictionCol='pred', rawPredictionCol='pred_raw')
    rf_model = rf.fit(trainDF)
    rf_result = rf_model.transform(testDF)

    area_under_pr = evaluator1.evaluate(rf_result)
    f1_score = evaluator2.evaluate(rf_result)
    accuracy = evaluator3.evaluate(rf_result)

    print("")
    print("########################################################################")
    print("RANDOM FOREST RESULTS")
    print("Area under PR curve: " + str(area_under_pr))
    print("F1 score = %g" % f1_score)
    print("Accuracy = %g" % accuracy)
    print("########################################################################")


    # ****************************************************************************** #
    # Run Gradient-Boosted Trees Classification.                                     #
    # ****************************************************************************** #

    gbt = GBTClassifier(featuresCol='features', labelCol='label', predictionCol='pred', maxIter=10)
    gbt_model = gbt.fit(trainDF)
    gbt_result = gbt_model.transform(testDF)

    f1_score = evaluator2.evaluate(gbt_result)
    accuracy = evaluator3.evaluate(gbt_result)

    print("")
    print("########################################################################")
    print("GRADIENT-BOOSTED TREE RESULTS")
    print("F1 score = %g" % f1_score)
    print("Accuracy = %g" % accuracy)
    print("########################################################################")

    print(lr.explainParams())

    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.aggregationDepth, [2, 3, 4])
                 .addGrid(lr.maxIter, [10, 20, 50])
                 .addGrid(lr.standardization, [True, False])
                 .addGrid(lr.threshold, [0.1, 0.5, 0.7])
                 .build())

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator2, numFolds=5)

    # Run cross validations.
    lr_cvModel = cv.fit(trainDF)
    lr_cvResult = lr_cvModel.transform(testDF)
    f1_score = evaluator2.evaluate(lr_cvResult)

    print("")
    print("########################################################################")
    print("5-FOLD CV LOGISTIC REGRESSION RESULTS")
    print("F1 score = %g" % f1_score)
    print("########################################################################")

    lr_cvResult.show(5)

    lr_finalResult = lr_cvModel.transform(to_predictDF)
    lr_finalResult = lr_finalResult.withColumnRenamed('pred', 'category')
    lr_finalResult = lr_finalResult.withColumn('category', lr_finalResult["category"].cast(IntegerType()))
    lr_finalResult.select("id", "category").write.csv("predictions.csv", header=True)

    spark.stop()









