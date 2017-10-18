package com.hyr.sparkml.als;

import java.io.Serializable;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

/**
 * @category ALS-WR
 * @author huangyueran
 *
 */
public class JavaALSExampleByMl {

	public static class Rating implements Serializable {
		// 0::2::3::1424380312
		private int userId; // 0
		private int movieId; // 2
		private float rating; // 3
		private long timestamp; // 1424380312

		public Rating() {
		}

		public Rating(int userId, int movieId, float rating, long timestamp) {
			this.userId = userId;
			this.movieId = movieId;
			this.rating = rating;
			this.timestamp = timestamp;
		}

		public int getUserId() {
			return userId;
		}

		public int getMovieId() {
			return movieId;
		}

		public float getRating() {
			return rating;
		}

		public long getTimestamp() {
			return timestamp;
		}

		public static Rating parseRating(String str) {
			String[] fields = str.split("::");
			if (fields.length != 4) {
				throw new IllegalArgumentException("Each line must contain 4 fields");
			}
			int userId = Integer.parseInt(fields[0]);
			int movieId = Integer.parseInt(fields[1]);
			float rating = Float.parseFloat(fields[2]);
			long timestamp = Long.parseLong(fields[3]);
			return new Rating(userId, movieId, rating, timestamp);
		}
	}

	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.WARN);
		SparkConf conf = new SparkConf().setAppName("JavaALSExample").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(jsc);

		JavaRDD<Rating> ratingsRDD = jsc.textFile("data/sample_movielens_ratings.txt")
				.map(new Function<String, Rating>() {
					public Rating call(String str) {
						return Rating.parseRating(str);
					}
				});
		DataFrame ratings = sqlContext.createDataFrame(ratingsRDD, Rating.class);
		DataFrame[] splits = ratings.randomSplit(new double[] { 0.8, 0.2 }); // //对数据进行分割，80%为训练样例，剩下的为测试样例。
		DataFrame training = splits[0];
		DataFrame test = splits[1];

		// Build the recommendation model using ALS on the training data
		ALS als = new ALS().setMaxIter(5) // 设置迭代次数
				.setRegParam(0.01) // //正则化参数，使每次迭代平滑一些，此数据集取0.1好像错误率低一些。
				.setUserCol("userId").setItemCol("movieId")
				.setRatingCol("rating");
		ALSModel model = als.fit(training); // //调用算法开始训练
		
		
		DataFrame itemFactors = model.itemFactors();
		itemFactors.show(1500);
		DataFrame userFactors = model.userFactors();
		userFactors.show();
		
		// Evaluate the model by computing the RMSE on the test data
		DataFrame rawPredictions = model.transform(test); //对测试数据进行预测
		DataFrame predictions = rawPredictions
				.withColumn("rating", rawPredictions.col("rating").cast(DataTypes.DoubleType))
				.withColumn("prediction", rawPredictions.col("prediction").cast(DataTypes.DoubleType));

		RegressionEvaluator evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating")
				.setPredictionCol("prediction");
		Double rmse = evaluator.evaluate(predictions);
		System.out.println("Root-mean-square error = " + rmse); // 均方根误差;
		
		jsc.stop();
	}
}
