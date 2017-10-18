package com.hyr.sparkml.als;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.kafka.KafkaUtils;

import kafka.serializer.StringDecoder;
import scala.Tuple2;

/**
 * @category 基于Spark-streaming、kafka的实时推荐模板DEMO 原系统中包含商城项目、logback、flume、hadoop
 * The real time recommendation template DEMO based on Spark-streaming and Kafka contains the mall project, logback, flume and Hadoop in the original system
 * @author huangyueran
 *
 */
public final class SparkALSByStreaming {

	//	基于Hadoop、Flume、Kafka、spark-streaming、logback、商城系统的实时推荐系统DEMO
	//	Real time recommendation system DEMO based on Hadoop, Flume, Kafka, spark-streaming, logback and mall system
	//	商城系统采集的数据集格式 Data Format:
	//	用户ID，商品ID，用户行为评分，时间戳
	//	UserID,ItemId,Rating,TimeStamp
	//	53,1286513,9,1508221762
	//	53,1172348420,9,1508221762
	//	53,1179495514,12,1508221762
	//	53,1184890730,3,1508221762
	//	53,1210793742,159,1508221762
	//	53,1215837445,9,1508221762
	
	public static void main(String[] args) {
		System.setProperty("HADOOP_USER_NAME", "root"); // 设置权限用户
		
		SparkConf sparkConf = new SparkConf().setAppName("JavaKafkaDirectWordCount").setMaster("local[1]");

		final JavaStreamingContext jssc = new JavaStreamingContext(sparkConf, Durations.seconds(6));

		Map<String, String> kafkaParams = new HashMap<String, String>(); // key是topic名称,value是线程数量
		kafkaParams.put("metadata.broker.list", "master:9092,slave1:9092,slave2:9092"); // 指定broker在哪
		HashSet<String> topicsSet = new HashSet<String>();
		topicsSet.add("taotao-server-recommend-logs"); // 指定操作的topic

		// Create direct kafka stream with brokers and topics
		// createDirectStream()
		JavaPairInputDStream<String, String> messages = KafkaUtils.createDirectStream(jssc, String.class, String.class,
				StringDecoder.class, StringDecoder.class, kafkaParams, topicsSet);

		JavaDStream<String> lines = messages.map(new Function<Tuple2<String, String>, String>() {
			@Override
			public String call(Tuple2<String, String> tuple2) {
				return tuple2._2();
			}
		});

		JavaDStream<Rating> ratingsStream = lines.map(new Function<String, Rating>() {
			public Rating call(String s) {
				String[] sarray = StringUtils.split(StringUtils.trim(s), ",");
				return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
						Double.parseDouble(sarray[2]));
			}
		});

		// 进行流推荐计算
		ratingsStream.foreachRDD(new Function<JavaRDD<Rating>, Void>() {

			@Override
			public Void call(JavaRDD<Rating> ratings) throws Exception {
				// TODO 获取到原始的数据集
				SparkContext sc = ratings.context();

				RDD<String> textFileRDD = sc.textFile("hdfs://master:8020/flume/logs", 3); // 读取原始数据集文件
				JavaRDD<String> originalTextFile = textFileRDD.toJavaRDD();

				final JavaRDD<Rating> originaldatas = originalTextFile.map(new Function<String, Rating>() {
					public Rating call(String s) {
						String[] sarray = StringUtils.split(StringUtils.trim(s), ",");
						return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
								Double.parseDouble(sarray[2]));
					}
				});
				System.out.println("====================");
				System.out.println("originalTextFile count:" + originalTextFile.count()); // HDFS中已经存储的原始用户行为日志数据
				System.out.println("====================");

				// TODO 将原始数据集和新的用户行为数据进行合并
				JavaRDD<Rating> calculations = originaldatas.union(ratings);

				System.out.println("计算总数:" + calculations.count());

				// Build the recommendation model using ALS
				int rank = 10; // 模型中隐语义因子的个数
				int numIterations = 6; // 训练次数

				// 得到训练模型
				if (null != ratings && !ratings.isEmpty()) { // 如果有用户行为数据
					MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(calculations), rank, numIterations, 0.01);
					// TODO 判断文件是否存在,如果存在 删除文件目录
					Configuration hadoopConfiguration = sc.hadoopConfiguration();
					hadoopConfiguration.set("fs.defaultFS", "hdfs://master:8020");
					FileSystem fs = FileSystem.get(hadoopConfiguration);
					Path outpath = new Path("/spark-als/model");
					if (fs.exists(outpath)) {
						//System.out.println("########### 删除"+outpath.getName()+" ###########");
						fs.delete(outpath, true);
					}
					
					// 保存model
					model.save(sc, "hdfs://master:8020/spark-als/model");
					// TODO 读取model
					//MatrixFactorizationModel modelLoad = MatrixFactorizationModel.load(sc, "hdfs://master:8020/spark-als/model");

					// 为指定用户推荐10个商品(电影)
					Rating[] recommendProducts = model.recommendProducts(53, 10);
					for (Rating r : recommendProducts) {
						System.out.println(r.toString());
					}
				}

				return null;
			}
		});

		// ==========================================================================================

		jssc.start();
		jssc.awaitTermination();

		// jssc.stop();
		// jssc.close();
	}

}
