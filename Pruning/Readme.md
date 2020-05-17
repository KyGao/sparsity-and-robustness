# pruning-recapitulation

merging some articles about network pruning 
original articals:
1. [闲话模型压缩之网络剪枝](https://blog.csdn.net/ariesjzj/article/details/100621397)(2019.9)
2. [深度学习模型压缩方法综述](https://blog.csdn.net/wspba/java/article/details/75675554)(2017.7)

### Saliency-based（贪心法，按重要性排序）
> smaller-norm-less-important 准则

2016~ [Pruning Filters for Efficient ConvNets](https://arxiv.org/pdf/1608.08710.pdf)
2016年的经典论文，评判标准很简单，就是靠weight：对于一个filter，其中所有weight的绝对值求和，值低的filter裁掉。因为filter裁掉意味着对应的feature map也被裁掉，所以下一层对应的kernel也被裁掉。但是如果训练出来的权重不稀疏，即 variance 很小，常用的方法就是在训练时的 loss 中加 regularizer 使权重分布稀疏化。

2015-16~ *Learning Structured Sparsity in Deep Neural Networks*, *Sparse Convolutional Neural Networks*
属于structured pruning，表明了想获得结构化的稀疏权重，常用 Group Lasso来得到

2017~ *Learning Efficient Neural Networks Trough Network Slimming*
用BN层的 scaling factor 作为重要性度量，在BN层加入channel-wise scaling factor并加L1正则项使之稀疏，剪掉scaling factor值小的部分对应的权重

2016~ [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)
一般来说，像ReLU这样的激活函数本身会倾向于产生稀疏的activation，这样就不需要外力（正则项）来使它变得稀疏。这篇文章就提出采用Average Percentage of Zeros，即APoZ来衡量activation的重要性，用每一个filter中激活为0的值的数量，来作为评价一个filter是否重要的标准。作者发现在VGG-16中，有631个filter的APoZ超过了90%，也就说明了网络中存在大量的冗余。但是作者仅在最后一个卷积层和全连接层上进行了实验，因此该方法在实际中的效果很难保证

2018~*Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration*
讨论了magnitude-based方法的前提与局限（即需要其范数值方差大，且最小值接近于0）。它提出了 FPGM（Filter Pruning via Geometric Median）方法。其基本思想是基于geometric median来去除冗余的参数。

### 考虑了参数裁剪对loss的影响
1900s~ *OBD*, *OBS*
始祖级别，计算近似Hessian矩阵比较费时

2017~ [Pruning Convolutional Neural Networks for Resource Efficient Transfer Learning](https://arxiv.org/abs/1611.06440)

作者将裁剪问题当做一个组合优化问题：从众多的权重参数中选择一个最优的组合B，使得被裁剪的模型的代价函数的损失衰减最小。这也是OBD的思路，但是OBD的方法需要求二阶导数（Hessian矩阵），实现起来难度较大，而本文提出的Taylor expansion的方法可以很好的解决这个问题：基于Taylor展开，目标函数相对于activation的展开式中一阶项的绝对值作为剪枝指标。这样做还有一个优点就是，可以在训练的反向过程中将梯度记录下来，然后与feature map的activation直接相乘，可以达到边训练边裁剪的效果。

2018~ *SNIP: Single-shot Network Pruning based on Connection Sensitivity*
将归一化的目标函数相对于参数的导数绝对值作为剪枝指标

### 考虑了对特征输出的可重建性
即最小化裁剪后网络对于特征输出的重建误差，intuition是如果对当前层进行裁剪，然后如果它对后面输出还没啥影响，那说明裁掉的就是不重要的信息。

2017~ *ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression*
采用贪心法，最小化特征重建误差(*Feature reconstruction error*)来确定哪些channel需要裁剪

2017~ *Channel pruning for accelerating very deep neural networks*
采用LASSO Regression来建立最小化重建误差

2017～ *NISP: Pruning Networks using Neuron Importance Score Propagation*
提出只考虑后面一两层是不够的，于是提出NISP(Neuron importance score propagation)算法通过最小化分类网络倒数第二层的重建误差，并将重要性信息反向传播到前面以决定哪些channel需要裁剪

2018~ *Discrimination-aware Channel Pruning for Deep Neural Networks*
提出一种比较有意思的变体。它提出DCP（Discrimination-aware channel pruning）方法一方面在中间层添加额外的discrimination-aware loss（用以强化中间层的判别能力），另一方面也考虑特征重建误差的loss，综合两方面loss对于参数的梯度信息，决定哪些为需要被裁剪的channel。

### 考虑了参数间的相互关系（寻找全局最优解）
贪心算法的缺点就是只能找到局部最优解，因为它忽略了参数间的相互关系。那自然肯定会有一些方法会尝试考虑参数间的相互关系，试图找到全局更优解。

**基于熵的剪枝**

2017~ [An Entropy-based Pruning Method for CNN Compression](https://arxiv.org/pdf/1706.05791.pdf)

作者认为通过weight值的大小很难判定filter的重要性，通过这个来裁剪的话有可能裁掉一些有用的filter。因此作者提出了一种基于熵值的裁剪方式，利用熵值来判定filter的重要性。出发点是对于分类有决定作用的filter对于不同的输入会带来更多信息量。作者将每一层的输出通过一个Global average Pooling将feature map转换为一个长度为c（filter数量）的向量，对于n张图像可以得到一个n*c的矩阵，对于每一个filter，将它分为m个bin，统计每个bin的概率，然后计算它的熵值 利用熵值来判定filter的重要性，再对不重要的filter进行裁剪。

在retrain中，作者使用了这样的策略，即每裁剪完一层，通过少数几个迭代来恢复部分的性能，当所有层都裁剪完之后，再通过较多的迭代来恢复整体的性能，作者提出，在每一层裁剪过后只使用很少的训练步骤来恢复性能，能够有效的避免模型进入到局部最优。作者将自己的retrain方式与传统的finetuning方式进行比较，发现作者的方法能够有效的减少retrain的步骤，并也能达到不错的效果。（Soft Pruning）

**离散空间下的搜索**
2015~ *Structured Pruning of Deep Convolutional Neural Networks*
基于genetic algorithm与particle filter来进行网络的pruning
2017~ *N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning*
将网络的压缩分成两个阶段-layer removal和layer shrinkage，并利用强化学习（Reinforcement learning）分别得到两个阶段的策略。

**规划问题**
2019~ *Collaborative Channel Pruning for Deep Networks*
提出CCP（Collaborative channel pruning）方法，它考虑了channel间的依赖关系 ，将channel选取问题形式化为约束下的二次规划问题，再用SQP（Sequential quadratic programming）求解。

**Beyesian方法**
2017~ *Variational Dropout Sparsifies Deep Neural Networks*
提出了sparse variational droput。它对variational droput进行扩展使之可以对dropout rate进行调节，最终得到稀疏解从而起到裁剪模型的效果。

2019~ *Adaptive Network Sparsification with Dependent Variational Beta-Bernoulli Dropout*
在贝叶斯剪枝基础上提出的Beta-Bernoulli剪枝

**基于梯度的方法**
剪枝的数学表示是用L0-norm来表示是否剪掉，可是这样会导致目标不可微，从而无法用基于梯度的方法来求解

2017~ *Learning Sparse Neural Networks through L0 Regularization*
用一个连续的分布结合 hard-sigmoid recification去近似它，从而使目标函数平滑，这样便可以用基于梯度的方法求解

**基于聚类的方法**
一般地，对于压缩问题有一种方法就是采用聚类。如将图片中的颜色进行聚类，就可以减小其编码长度。类似地，在模型压缩中也可以用聚类的思想。

2018~ *SCSP: Spectral Clustering Filter Pruning with Soft Self-adaption Manners*, *Exploring Linear Relationship in Feature Map Subspace for ConvNets Compression*
分别用谱聚类和子空间聚类发掘filter和feature map中的相关信息，从而对参数进行简化压缩。

### 考虑了Sparsity Ratio，即每层剪多少
这里的sparsity ratio定义为层中为0参数所占比例，有些文章中也称为pruning rate等。从目标结构或者sparsity ratio的指定方式来说，按2018年论文《Rethinking the Value of Network Pruning》中的说法可分为预定义（predifined）和自动（automatic）两种方式。Predefined方法由人工指定每一层的比例进行裁剪，因此目标结构是提前确定。而automatic方法会根据所有的layer信息（即全局信息）由pruning算法确定每层裁剪比例，因此目标结构一开始并不确定。

**Predefined**
2019~ *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*
提出的EfficientNet将这种参数调节更进一步，提出compound scaling method将width, depth, resolution按特定比例一起调节。但它们调节参数都是针对网络中所有层的，粒度比较粗。显然，网络中不同层对于pruning的敏感（sensitivity）程度是不一样的，只有根据层的属性为每层设置最适合的sparsity ratio才是最优的，这种为每层专设的称为local sparsity，相对地前面那种就称为global sparsity

**Autumatic**
2019~ *Play and Prune: Adaptive Filter Pruning for Deep Model Compression*
将pruning问题建模成min-max优化问题，然后通过两个模块交替迭代分别进行裁剪和通过调节pruning rate控制精度损失

2018~ *ADC: Automated Deep Compression and Acceleration with Reinforcement Learning*
提出ADC（Automated deep compression）方法，根据不同需求（如保证精度还是限制计算量），利用强化学习来学习每一层最优的sparsity ratio

2018~ *“Learning-Compression” Algorithms for Neural Net Pruning*
提出Learning和Compression两步交替优化的pruning方法，在Compression操作中，通过将原参数向约束表示的可行集投影来自动找到每层的最优sparsity ratio。


### 经典的剪枝原理性讨论
2019~ *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*
提出The Lottery Ticket Hypothesis，即一个随机初始化，密集的网络包含一个子网络，这个子网络如果沿用原网络的权重初始化，在至多同样迭代次数训练后就可以比肩原网络的测试精度。同时它还给出了找这种子网络结构的方法。文章认为这个子结构和它的初始值对训练的有效性至关重要，它们被称为『winning logttery tickets』。

2018~ *Rethinking the Value of Network Pruning*
提出不仅over-parameterization对于训练不是重要的，而且从原网络中重用其权重也未必是很好的选择，它可能会使裁剪后的模型陷入局部最小。如果原网络的权重或其初始值不重要的话，那剩下最重要的就是pruning后的网络结构了。换句话说，某种意义上来说，pruning即是neural architecture search（NAS），只是由于 它只涉及层的维度，搜索空间相比小一些。但这也是它的优点，搜索空间小了自然搜索就高效了。

### 保留模型capacity的一种方法
Pruning按最初的字面意思理解就是给模型做减法。之前的主流pruning方法中，被裁剪的部分一般也就直接丢弃不会再拿回来了，即模型的capacity在iterative pruning的过程中不断减少。这样的话，一旦有参数被不适当地裁剪掉，便无法被恢复。而这两年，**学界正在尝试在模型压缩过程中保留被裁剪部分能力或者扩充能力的方法**。

2018~ *Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks*
提出SFP（Soft filter pruning）让被裁剪的filter在训练中仍能被更新，这样它仍有机会被恢复回来

2019~ *Asymptotic Soft Pruning for Deep Convolutional Neural Networks*
基于SFP提出的剪枝方法，就是动态改变每次影响的权重数量

2016~ *Dynamic Network Surgery for Efficient DNNs*
在pruning的基础上加了splicing操作，避免不合适的pruning带来的影响

2017~ *Morphnet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks*
迭代地进行shrink和expand的操作。 
