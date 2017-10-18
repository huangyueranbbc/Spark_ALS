#Spark-ALS  
简介  
  
ALS是alternating least squares的缩写 , 意为交替最小二乘法；而ALS-WR是alternating-least-squares with weighted-λ -regularization的缩写，意为加权正则化交替最小二乘法。该方法常用于基于矩阵分解的推荐系统中。例如：将用户(user)对商品(item)的评分矩阵分解为两个矩阵：一个是用户对商品隐含特征的偏好矩阵，另一个是商品所包含的隐含特征的矩阵。在这个矩阵分解的过程中，评分缺失项得到了填充，也就是说我们可以基于这个填充的评分来给用户最商品推荐了。    
ALS is the abbreviation of squares alternating least, meaning the alternating least squares method; and the ALS-WR is alternating-least-squares with weighted- lambda -regularization acronym, meaning weighted regularized alternating least squares method. This method is often used in recommender systems based on matrix factorization. For example, the user (user) score matrix of item is decomposed into two matrices: one is the user preference matrix for the implicit features of the commodity, and the other is the matrix of the implied features of the commodity. In the process of decomposing the matrix, the score missing is filled, that is, we can give the user the most recommended commodity based on the filled score.

ALS-WR算法，简单地说就是：  
（数据格式为：userId, itemId, rating, timestamp ）  
1 对每个userId随机初始化N（10）个factor值，由这些值影响userId的权重。  
2 对每个itemId也随机初始化N（10）个factor值。  
3 固定userId，从userFactors矩阵和rating矩阵中分解出itemFactors矩阵。即[Item Factors Matrix] = [User Factors Matrix]^-1 * [Rating Matrix].  
4 固定itemId，从itemFactors矩阵和rating矩阵中分解出userFactors矩阵。即[User Factors Matrix] = [Item Factors Matrix]^-1 * [Rating Matrix].  
5 重复迭代第3，第4步，最后可以收敛到稳定的userFactors和itemFactors。  
6 对itemId进行推断就为userFactors * itemId = rating value；对userId进行推断就为itemFactors * userId = rating value。  

#SparkALSByStreaming.java  
基于Hadoop、Flume、Kafka、spark-streaming、logback、商城系统的实时推荐系统DEMO  
Real time recommendation system DEMO based on Hadoop, Flume, Kafka, spark-streaming, logback and mall system  
商城系统采集的数据集格式 Data Format:  
用户ID，商品ID，用户行为评分，时间戳  
UserID,ItemId,Rating,TimeStamp  
53,1286513,9,1508221762  
53,1172348420,9,1508221762  
53,1179495514,12,1508221762  
53,1184890730,3,1508221762  
53,1210793742,159,1508221762  
53,1215837445,9,1508221762  