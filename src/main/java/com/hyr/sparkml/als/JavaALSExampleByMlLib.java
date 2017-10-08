package com.hyr.sparkml.als;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;

/**
 * @category ALS
 * @author huangyueran
 *
 */
public class JavaALSExampleByMlLib {

	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.WARN);
		SparkConf conf = new SparkConf().setAppName("JavaALSExample").setMaster("local[4]");
		JavaSparkContext jsc = new JavaSparkContext(conf);

		JavaRDD<String> data = jsc.textFile("data/sample_movielens_ratings.txt");

		JavaRDD<Rating> ratings = data.map(new Function<String, Rating>() {
			public Rating call(String s) {
				String[] sarray = StringUtils.split(StringUtils.trim(s), "::");
				return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
						Double.parseDouble(sarray[2]));
			}
		});

		// Build the recommendation model using ALS
		int rank = 10;
		int numIterations = 6;
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

		// Evaluate the model on rating data
		JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(new Function<Rating, Tuple2<Object, Object>>() {
			public Tuple2<Object, Object> call(Rating r) {
				return new Tuple2<Object, Object>(r.user(), r.product());
			}
		});

		// 预测的评分
		JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD
				.fromJavaRDD(model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD()
						.map(new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
							public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
								return new Tuple2<Tuple2<Integer, Integer>, Double>(
										new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
							}
						}));

		JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD
				.fromJavaRDD(ratings.map(new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
					public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
						return new Tuple2<Tuple2<Integer, Integer>, Double>(
								new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
					}
				})).join(predictions);

		// 得到按照用户ID排序后的评分列表 key:用户id
		JavaPairRDD<Integer, Tuple2<Integer, Double>> fromJavaRDD = JavaPairRDD.fromJavaRDD(ratesAndPreds.map(
				new Function<Tuple2<Tuple2<Integer, Integer>, Tuple2<Double, Double>>, Tuple2<Integer, Tuple2<Integer, Double>>>() {

					@Override
					public Tuple2<Integer, Tuple2<Integer, Double>> call(
							Tuple2<Tuple2<Integer, Integer>, Tuple2<Double, Double>> t) throws Exception {
						// TODO Auto-generated method stub
						return new Tuple2<Integer, Tuple2<Integer, Double>>(t._1._1,
								new Tuple2<Integer, Double>(t._1._2, t._2._2));
					}
				})).sortByKey(false);
		
//		List<Tuple2<Integer,Tuple2<Integer,Double>>> list = fromJavaRDD.collect();
//		for(Tuple2<Integer,Tuple2<Integer,Double>> t:list){
//			System.out.println(t._1+":"+t._2._1+"===="+t._2._2);
//		}

		JavaRDD<Tuple2<Double, Double>> ratesAndPredsValues = ratesAndPreds.values();

		double MSE = JavaDoubleRDD.fromRDD(ratesAndPredsValues.map(new Function<Tuple2<Double, Double>, Object>() {
			public Object call(Tuple2<Double, Double> pair) {
				Double err = pair._1() - pair._2();
				return err * err;
			}
		}).rdd()).mean();

		try {
			FileUtils.deleteDirectory(new File("result"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		ratesAndPreds.repartition(1).saveAsTextFile("result/ratesAndPreds.txt");

		//为指定用户推荐10个商品(电影)
		Rating[] recommendProducts = model.recommendProducts(2, 10);
		for(Rating r:recommendProducts){
			System.out.println(r.toString());
		}
		
		// 为所有用户推荐TOP N个物品
		//model.recommendUsersForProducts(10);
		
		// 为所有物品推荐TOP N个用户
		//model.recommendProductsForUsers(10)
		
		model.userFeatures().saveAsTextFile("result/userFea.txt");
		model.productFeatures().saveAsTextFile("result/productFea.txt");
		System.out.println("Mean Squared Error = " + MSE);

	}

}
