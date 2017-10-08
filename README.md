#Spark-ALS  
简介  
  
ALS是alternating least squares的缩写 , 意为交替最小二乘法；而ALS-WR是alternating-least-squares with weighted-λ -regularization的缩写，意为加权正则化交替最小二乘法。该方法常用于基于矩阵分解的推荐系统中。例如：将用户(user)对商品(item)的评分矩阵分解为两个矩阵：一个是用户对商品隐含特征的偏好矩阵，另一个是商品所包含的隐含特征的矩阵。在这个矩阵分解的过程中，评分缺失项得到了填充，也就是说我们可以基于这个填充的评分来给用户最商品推荐了。    

ALS-WR算法，简单地说就是：  
（数据格式为：userId, itemId, rating, timestamp ）  
1 对每个userId随机初始化N（10）个factor值，由这些值影响userId的权重。  
2 对每个itemId也随机初始化N（10）个factor值。  
3 固定userId，从userFactors矩阵和rating矩阵中分解出itemFactors矩阵。即[Item Factors Matrix] = [User Factors Matrix]^-1 * [Rating Matrix].  
4 固定itemId，从itemFactors矩阵和rating矩阵中分解出userFactors矩阵。即[User Factors Matrix] = [Item Factors Matrix]^-1 * [Rating Matrix].  
5 重复迭代第3，第4步，最后可以收敛到稳定的userFactors和itemFactors。  
6 对itemId进行推断就为userFactors * itemId = rating value；对userId进行推断就为itemFactors * userId = rating value。  
