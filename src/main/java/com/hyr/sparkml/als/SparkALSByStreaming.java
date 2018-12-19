package com.hyr.sparkml.als;

import java.util.Arrays;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

/**
 * @author huangyueran
 * @category 基于Spark-streaming、kafka的实时推荐模板DEMO 原系统中包含商城项目、logback、flume、hadoop
 * The real time recommendation template DEMO based on Spark-streaming and Kafka contains the mall project, logback, flume and Hadoop in the original system
 */
public final class SparkALSByStreaming {

    private static final Logger log = LoggerFactory.getLogger(SparkALSByStreaming.class);

    private static final String KAFKA_ADDR = "middleware:9092";
    private static final String TOPIC = "RECOMMEND_TOPIC";
    private static final String HDFS_ADDR = "hdfs://middleware:9000";

    private static final String MODEL_PATH = "/spark-als/model";


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
        kafkaParams.put("metadata.broker.list", KAFKA_ADDR); // 指定broker在哪
        HashSet<String> topicsSet = new HashSet<String>();
        topicsSet.add(TOPIC); // 指定操作的topic

        // Create direct kafka stream with brokers and topics
        // createDirectStream()
        JavaPairInputDStream<String, String> messages = KafkaUtils.createDirectStream(jssc, String.class, String.class,
                StringDecoder.class, StringDecoder.class, kafkaParams, topicsSet);

        JavaDStream<String> lines = messages.map(new Function<Tuple2<String, String>, String>() {
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

            public Void call(JavaRDD<Rating> ratings) throws Exception {
                //  获取到原始的数据集
                SparkContext sc = ratings.context();

                RDD<String> textFileRDD = sc.textFile(HDFS_ADDR + "/flume/logs", 3); // 读取原始数据集文件
                JavaRDD<String> originalTextFile = textFileRDD.toJavaRDD();

                final JavaRDD<Rating> originaldatas = originalTextFile.map(new Function<String, Rating>() {
                    public Rating call(String s) {
                        String[] sarray = StringUtils.split(StringUtils.trim(s), ",");
                        return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Double.parseDouble(sarray[2]));
                    }
                });
                log.info("========================================");
                log.info("Original TextFile Count:{}", originalTextFile.count()); // HDFS中已经存储的原始用户行为日志数据
                log.info("========================================");

                //  将原始数据集和新的用户行为数据进行合并
                JavaRDD<Rating> calculations = originaldatas.union(ratings);

                log.info("Calc Count:{}", calculations.count());

                // Build the recommendation model using ALS
                int rank = 10; // 模型中隐语义因子的个数
                int numIterations = 6; // 训练次数

                // 得到训练模型
                if (!ratings.isEmpty()) { // 如果有用户行为数据
                    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(calculations), rank, numIterations, 0.01);
                    //  判断文件是否存在,如果存在 删除文件目录
                    Configuration hadoopConfiguration = sc.hadoopConfiguration();
                    hadoopConfiguration.set("fs.defaultFS", HDFS_ADDR);
                    FileSystem fs = FileSystem.get(hadoopConfiguration);
                    Path outpath = new Path(MODEL_PATH);
                    if (fs.exists(outpath)) {
                        log.info("########### 删除" + outpath.getName() + " ###########");
                        fs.delete(outpath, true);
                    }

                    // 保存model
                    model.save(sc, HDFS_ADDR + MODEL_PATH);

                    //  读取model
                    MatrixFactorizationModel modelLoad = MatrixFactorizationModel.load(sc, HDFS_ADDR + MODEL_PATH);
                    // 为指定用户推荐10个商品(电影)
                    for(int userId=0;userId<30;userId++){ // streaming_sample_movielens_ratings.txt
                        Rating[] recommendProducts = modelLoad.recommendProducts(userId, 10);
                        log.info("get recommend result:{}", Arrays.toString(recommendProducts));
                    }
                }

                return null;
            }
        });

        // ==========================================================================================

        jssc.start();
        jssc.awaitTermination();


        // Local Model
        try {
            Thread.sleep(10000000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        // jssc.stop();
        // jssc.close();
    }

}