- 2018-6-17 更新推荐方法接口

一个简单的电影推荐系统。  

## 1. Glimpse

模型移植至仓库[https://github.com/chengstone/movie_recommender](https://github.com/chengstone/movie_recommender)，原作者使用的是TensorFlow-1.0,本仓库得到作者允许后，使用PyTorch-0.4.0的移植版本。这里要感谢原作者对于框架部分的详细实现说明，使得我在复现过程中省去了很多麻烦。 

 模型使用了 **movieLens-1m** 的数据集进行训练，建模用户-电影-评分数据，欲实现给定用户和电影ID，预测用户对于电影的评分。在推荐系统领域，就是一个 **rating** 的任务。



## 2.模型概览

![](https://markdownfoto-1252952266.cos.ap-guangzhou.myqcloud.com/Github/model.001.jpeg)

**注**：图片来源原作者仓库  



模型基本就是使用双通道网络来实现推荐系统领域比较有效的基于 **rating** 的 **SVD++** 模型。这里简单讲下什么是SVD++

>**SVD++**：
>
>首先SVD++是SVD的推广。简单来说，SVD就是使用奇异值矩阵分解的方法，将推荐系统中的两个输入：用户矩阵和物品矩阵映射到隐藏空间，得到两个 dense 的隐藏向量，作为用户和物品的潜在喜好特征，之后对这两个向量进行点乘，获得最终的物品打分。那么++就是说，在考虑隐藏向量的时候，同时考虑一些辅助信息的作用，比如用户方面的用户个人信息等等。



## 3. 代码构成

在说明具体实现前，先来讲下代码的构成。  

代码很简单，一共5个文件(4个代码文件+1个数据文件)

- **data.p**: 保存了输入数据的pickle文件，加载完毕后是一个pandas(>=0.22.0)的DataFrame对象(如图)

  ![](https://markdownfoto-1252952266.cos.ap-guangzhou.myqcloud.com/Github/rec1.PNG)

- **dataset.py**: 包含`torch.utils.data.Dataset` 类，提供给 `torch.utils.data.Dataloader` 加载数据

- **model.py**: Pytorch对于上述模型的实现

- **main.py**: 主文件。提供模型训练等操作

- **recInterface.py**: 推荐方法接口，包含KNN等推荐方法



## 4. 模型实现

模型为用户-电影双通道。

- **用户通道**: 主要是对用户ID、性别、职业等属性进行特征抽取，主要用到了 `torch.nn.Embedding` 对这些索引值进行 look up。分别获得不同属性的特征，之后通过两层全连接综合这些属性特征，获得一个总的用户特征。
- **电影通道**: 一边同样使用look up获得电影id和类型的嵌入。一边则使用文本卷积网络提取电影名的特征，最后用全连接获得最终的电影特征。  

最后对两个特征进行点积操作获得评分。    

可以看到这里用到了很多次的`Embedding`，所以在讲具体的代码实现前，先声明两个模型使用到的词典，来保存不同 `Embedding` 需要的最大值参数。

```python
user_max_dict={
    'uid':6041,  # 用户ID的最大值
    'gender':2,	 # 0/1 表示男女
    'age':7,     # 7个年龄段
    'job':21     # 21个工作分类
}

movie_max_dict={
    'mid':3953,  # 电影ID的最大值
    'mtype':18,  # 电影类型的最大值
    'mword':5215 # 电影名的词典term总数
}
```

同时，模型用到了文本卷积网络。原作者在实现的过程中，用了4个不同的卷积核，所以定义一下卷积核参数

```python
convParams={
    'kernel_sizes':[2,3,4,5]
}
```

### 4.1 用户通道

用户通道主要进行连个操作： Embedding 和 全连接。所以在自定义的网络类中，预先定义如下网络层：

```python
# --------------------------------- user channel --------------------------------------------
# user embeddings
self.embedding_uid = nn.Embedding(user_max_dict['uid'], embed_dim)
self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim // 2)
self.embedding_age = nn.Embedding(user_max_dict['age'], embed_dim // 2)
self.embedding_job = nn.Embedding(user_max_dict['job'], embed_dim // 2)

# user embedding to fc: the first dense layer
self.fc_uid = nn.Linear(embed_dim, embed_dim)
self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
self.fc_age = nn.Linear(embed_dim // 2, embed_dim)
self.fc_job = nn.Linear(embed_dim // 2, embed_dim)

# concat embeddings to fc: the second dense layer
self.fc_user_combine = nn.Linear(4 * embed_dim, fc_size)
```

最后一步连接所有的 attribute embedding，做一个全连接

### 4.2 电影通道

**part 1: attribute  embedding**  

操作和用户通道类似:

```python
# movie embeddings
self.embedding_mid = nn.Embedding(movie_max_dict['mid'], embed_dim)  # normally 32
self.embedding_mtype_sum = nn.EmbeddingBag(movie_max_dict['mtype'], embed_dim, mode='sum')

self.fc_mid = nn.Linear(embed_dim, embed_dim)
self.fc_mtype = nn.Linear(embed_dim, embed_dim)

# movie embedding to fc
self.fc_mid_mtype = nn.Linear(embed_dim * 2, fc_size)
```

**part2: textCNN**  

文本卷积网络，根据卷积核个数，定义几个卷积子网络即可：

```python
# wordlist to embedding matrix B x L x D  L=15 15 words
self.embedding_mwords = nn.Embedding(movie_max_dict['mword'], embed_dim)

# input word vector matrix is B x 15 x 32
# load text_CNN params
kernel_sizes = convParams['kernel_sizes']
# 8 kernel, stride=1,padding=0, kernel_sizes=[2x32, 3x32, 4x32, 5x32]
self.Convs_text = [nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(15 - k + 1, 1), stride=(1, 1))
).to(device) for k in kernel_sizes]

# movie channel concat
self.fc_movie_combine = nn.Linear(embed_dim * 2 + 8 * len(kernel_sizes), fc_size)  # tanh
```



## 5. 待续

可能会有改动补充，待续.