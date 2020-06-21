#Porównanie wybranych metod klasyfikacji binarnej w problemie prognozowania ocen filmów
#Wybrane metody: drzewo decyzyjne, regresja logistyczna
#Joanna Zając

from pyspark.sql import SparkSession
spark= SparkSession.builder.getOrCreate()
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from datetime import datetime
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import numpy as np

#import danych
movies=spark.read.csv("movies.csv",header=True).select(['movieId','title','genres'])
ratings=spark.read.csv("ratings.csv", header=True).select(['movieId','userId','rating'])
tags=spark.read.csv("tags.csv", header=True).select(['movieId','userId','tag'])
tag_relev=spark.read.csv("genome-scores.csv", header=True).select(['movieId','tagId','relevance'])
tagsIds=spark.read.csv("genome-tags.csv", header=True).select(['tagId','tag'])
tags=spark.read.csv("tags.csv", header=True).select(['movieId','userId','tag'])

#łączenie danych w jedną tabelę
md=ratings.join(movies, on='movieId', how="leftouter")
md=md.join(tags, on=['movieId','userId'], how="leftouter")
md=md.join(tagsIds, on='tag', how="leftouter")
model_data=md.join(tag_relev, on=['movieId','tagId'], how="leftouter")
model_data.printSchema()

#zmiana typu danych, utowrzenie label
model_data=model_data.withColumn("rating",model_data.rating.cast("double"))
model_data=model_data.withColumn("movieId",model_data.movieId.cast("integer"))
model_data=model_data.withColumn("tagId",model_data.tagId.cast("integer"))
model_data=model_data.withColumn("userId",model_data.userId.cast("integer"))
model_data=model_data.withColumn("relevance",model_data.relevance.cast("double"))
model_data=model_data.withColumn("is_good", model_data.rating >= 4)
model_data=model_data.withColumn("is_good", model_data.is_good.cast("double"))
model_data=model_data.withColumnRenamed('is_good','label')

#przypisanie kolumn 
cols = model_data.columns
model_data.printSchema()
model_data.show(30)

#usuwanie brakujących danych
model_data = model_data.filter("movieId is not NULL and userId is not NULL and label is not NULL and genres is not NULL and tagId is not NULL and relevance is not NULL")
model_data.show(30)
'''
#dane statystyczne
model_data.groupBy("rating").count().sort(desc("rating")).show()
model_data.groupBy("label").count().show()
model_data.describe("rating","relevance").show()
n=model_data.count()
uniqueusers=model_data.dropDuplicates(subset=['userId']).count()
uniquemovies=model_data.dropDuplicates(subset=['movieId']).count()
print("Liczba użytkowników:", uniqueusers, "\nLiczba filmów:",uniquemovies, "\nLiczba wierszy:", n)
model_data.groupBy("tag").count().sort(desc("count")).show()
model_data.groupBy("genres").count().sort(desc("count")).show()
'''
#kodowanie
genres_indexer= StringIndexer(inputCol="genres", outputCol="genres_index")
genres_encoder=OneHotEncoder(inputCol="genres_index", outputCol="genres_fact")
tag_indexer= StringIndexer(inputCol="tag", outputCol="tag_index")
tag_encoder=OneHotEncoder(inputCol="tag_index", outputCol="tag_fact")
vec_assembler = VectorAssembler(inputCols=["movieId", "tagId","tag_fact","userId","genres_fact","relevance"], outputCol="features")

#tworzenie potoku
pipeline=Pipeline(stages=[genres_indexer, genres_encoder,tag_indexer, tag_encoder, vec_assembler])

#dopasowanie modelu
pipelineModel = pipeline.fit(model_data)
model_data = pipelineModel.transform(model_data)
selectedCols = ['features'] + cols
model_data = model_data.select(selectedCols)
model_data.printSchema()

#podział na zbiór treningowy i testowy
train, test = model_data.randomSplit([0.7, 0.3], seed = 2018)

#statystyki danych w zbiorach testowym i treningowym
print("Liczba danych w zbiorze treningowym:", train.count())
print("Liczba danych w zbiorze testowym:", test.count())
print("Struktura w zbiorze treningowym:")
train.groupBy("label").count().show()
print("Struktura w zbiorze testowym:")
test.groupBy("label").count().show()

#drzewo decyzyjne
tstart=datetime.now()
dt=DecisionTreeClassifier(featuresCol="features",labelCol="label")
dtmodel=dt.fit(train)
predictions = dtmodel.transform(test)
tend=datetime.now()
predictions.select('label', 'prediction', 'probability').show(10)
print("dt time", tend-tstart)


#obliczenie wartosci tp,tn,fp,fn
tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Total", predictions.count())
#macierz pomyłek
metrics = MulticlassMetrics(predictions.select("prediction", "label").rdd)
print(metrics.confusionMatrix())
#oceny modelu: recall, precision, specificity - sposób pierwszy - wprost ze wzorów
r = float(tp)/(tp + fn)
print("recall","{:.4f}".format(r))
p = float(tp) / (tp + fp)
print("precision","{:.4f}".format(p))
s=tn/(tn+fp)
print("specificity","{:.4f}".format(s))
#oceny modelu: weightedPrecision, accuracy,f1 - sposób drugi - wbudowane ewaluatory
ev1=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="accuracy")
ev2=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="weightedPrecision")
ev3=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
acc=ev1.evaluate(predictions)
print(ev1.getMetricName(), "{:.4f}".format(acc))
error=1-acc
print("error", "{:.4f}".format(1-acc))
print(ev2.getMetricName(), "{:.4f}".format(ev2.evaluate(predictions)))
print(ev3.getMetricName(), "{:.4f}".format(ev3.evaluate(predictions)))
#pole pod krzywą ROC
evaluator = BinaryClassificationEvaluator(labelCol='label',rawPredictionCol="rawPrediction")
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("area under ROC curve {:.4f}".format(auroc))



#regresja logistyczna


tstart=datetime.now()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

#walidacja krzyżowa
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Import the tuning submodule

#siatka parametrów
grid = tune.ParamGridBuilder()

# dodanie hyperparamrtów
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
grid = grid.build()

# Krosswalidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

lrmodel = lr.fit(train)

#najlepszy lr
print(lrmodel)



predictions = lrmodel.transform(test)
tend=datetime.now()
predictions.select('label', 'prediction', 'probability').show(10)
print("Czas modelu lr:", tend - tstart)

#obliczenie wartosci tp,tn,fp,fn
tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Total", predictions.count())

#macierz pomyłek
metrics = MulticlassMetrics(predictions.select("prediction", "label").rdd)
print(metrics.confusionMatrix())

#oceny modelu - jak poprzednio
r = float(tp)/(tp + fn)
print("recall","{:.4f}".format(r))
p = float(tp) / (tp + fp)
print("precision","{:.4f}".format(p))
s=tn/(tn+fp)
print("specificity","{:.4f}".format(s))

ev1=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="accuracy")
ev2=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="weightedPrecision")
ev3=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
acc=ev1.evaluate(predictions)
print(ev1.getMetricName(), "{:.4f}".format(acc))
error=1-acc
print("error", "{:.4f}".format(1-acc))
print(ev2.getMetricName(), "{:.4f}".format(ev2.evaluate(predictions)))
print(ev3.getMetricName(), "{:.4f}".format(ev3.evaluate(predictions)))

evaluator = BinaryClassificationEvaluator(labelCol='label',rawPredictionCol="rawPrediction")
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("area under ROC curve {:.4f}".format(auroc))