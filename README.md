## Chinese Intent Match 2018-11

#### 1.preprocess

prepare() 将按类文件保存的数据原地去重，augment() 进行数据增强

gather() 将数据汇总、打乱，保存为 (text, label) 格式，make_pair() 对每条数据

从同类、异类中抽样组合为正例、反例，保存为 (text1, text2, flag) 格式

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

vectorize() 和 vectorize_pair() 分别进行向量化，label2ind() 建立标签索引

#### 4.build

train 80% / dev 20% 划分，分别通过 dnn、cnn、rnn 构建匹配模型

#### 5.encode

定义模型的编码部分、按层名载入相应权重，对训练数据进行预编码

每类编码内部，孤立森林过滤离群点、k-means 保留中心点，提高匹配效率

#### 6.match

定义模型的匹配部分、按层名载入相应权重，读取缓存数据

predict() 实时交互，输入单句、清洗后进行预测，输出相似概率前 5 的语句

#### 7.eval

通过最近邻判决得到标签，test_pair()、test() 分别评估匹配、分类