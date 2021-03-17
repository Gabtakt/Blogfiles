---
title: AlphaJoin
typora-root-url: ../
---
# AlphaJoin: 连接顺序选择器

## 摘要

查询优化至今仍然是一个困难的问题，现存的DBMS经常错过一个好的连接顺序，识别一个好的连接顺序是提高数据库性能的关键，主要的挑战在于连接顺序的选择需要枚举的可能连接顺序，在一个较大的搜索空间内能提升找到更优解的可能性，但也会提升查询的时间开销。

受到AlphaGo的启发，提出一个叫**AlphaJoin**的方法，应用了AlphaGo中的**蒙特卡洛树搜索**，目前的研究表明该方法在性能上完全超过了PostgreSQL采用的最先进的方法。

## 介绍

数据库调优在提高DBMS性能上是很重要的一点，传统的启发式方法如动态规划，贪心算法，遗传算法和模拟退火算法等依赖于全局的静态信息，它们工作的时候没有学习历史的查询信息。面对最近ML在解决各种计算科学问题上的成功，很自然地就会想到用ML的方法来为表连接顺序选择进行优化。Marcus等人提出一个待验证阶段的选择顺序产生器ReJOIN，使用了强化学习的方法。目前的研究结果表明ReJOIN比PostgreSQL在产生选择顺序上做得更好，然而它跟传统优化器一样是依赖于**代价模型**的，代码模型是人为设计的，依赖于静态的信息，不会动态的改变。换言之，有可能选择出的方案代价最小，但真实执行时间并不是最小。Marcus等人提出一个名叫NEO的学习模型来生成高效的连接顺序，它使用真实执行时间而非代价模型，可以达到甚至超过传统优化器的优化效果。然而，这些方法都是基于简单搜索策略，这种策略搜索**不均衡**，有可能陷入局部最优解。

受到AlphaGo的启发，我们提出了采用蒙特卡洛树的搜索方案，主要思想是模拟许多连接顺序的可能在一个树结构上来进行搜索，具体来说，我们做了以下几点工作：

1. 第一次将MCTS用于学习和生成高效的连接顺序，设计了一个神经网络(Order Value Network,OVN)来预测给定查询计划的执行时间，并且在该网络内执行MCTS为查询计划打分，这是AlphaJoin 1.0。
2. 基于AlphaJoin 1.0，我们设计了一个自适应决策网络(Adaptive Decision Network,ADN)来选择最终的连接顺序是由OVN给出还是原始查询优化器给出。
3. 我们的实验结果证明了AlphaJoin能够产生有效的连接顺序，相对于NEO和PostgreSQL的传统优化器来说性能更好。

## 方法

在这一节中，我们首先描述两个编码方式(查询语句编码和查询计划编码)，然后对AlphaJoin进行一个综述。

###  Encodings

![图1](/images/AlphaJoin%20%E8%BF%9E%E6%8E%A5%E9%A1%BA%E5%BA%8F%E9%80%89%E6%8B%A9%E5%99%A8/image-20201217171517053.png)

**SQL-encoding** ：对SQL查询计划中的**表及属性信息**进行编码，每个查询计划可表示成两个部分，第一个部分将查询的连接信息以邻接矩阵进行保存，如上图中第1行第3列为1则表示表'A'和表'C'进行连接；第二个部分对包含的SQL谓词2属性进行简单的"ope-hot encoding"，例如B.a2为1则表示对表'B'的'a2'属性列有选择条件。

**Plan-encoding**：除了SQL=encoding外，我们还需要表示部分或完整的查询计划，每个编码和相应的查询计划之间是**一一对应**的，即计划编码是可编码和可解码的。在前人所做的工作ReJOIN中，使用了Huffman编码，这种方式难以表示稠密树，不能区分左右子树，也不能区分主动表和被动表。本文设计了一个查询计划编码，也包含两个部分（见上图）。

与SQL-encoding唯一不同之处在于，我们用查询顺序取代了原来的编码矩阵中的"1"，例如上图中第三行第五列为"4"，表示最先执行的表连接，即先连接表"C"和表"E"得到CE，然后第四行第三列为"3"，表示连接表"D"和表"C"，然而表"C"已经和表"E"连接过了，所以是将表"D"与连接结果CE进行连接，即D(CE)。随后，第一行第二列为"2"，表示对表"A"和表"B"进行连接得到AB。最后，第三行第一列为"1"，表示将表"C"与表"A"进行连接，然而表"C"和表"A"都连接过了，即表示将包含表"C"的连接结果与包含表"A"的连接结果进行连接，即(D(CE))(AB)。按照这个原则，解码信息包含了连接顺序与主动表和被动表的区分（行表示主动表，列表示被动表），解码出来的连接顺序是唯一的。这个编码方式类似于AlphaGO中的moves编码。

### AlphaJoin1.0

AlphaJoin1.0包含两个部分：Order Value Network（顺序价值网络）和MCTS（Monte Carlo tree Search,蒙特卡洛树搜索）。

#### Order Value Network

Order value network（OVN）是一种用来预测部分执行计划的最佳查询执行时间的深度神经网络，其架构如下图。

![图2](/images/AlphaJoin%20%E8%BF%9E%E6%8E%A5%E9%A1%BA%E5%BA%8F%E9%80%89%E6%8B%A9%E5%99%A8/image-20201221112542857.png)

它包含了一个输入层I，三个使用了*ReLU*作为激活函数的隐藏层H，和一个输出层O。因为这个神经网络的目标是估计一个具体的执行计划所花费的执行时间是快还是慢，所以输入数据是plan-encoding，输出是一个多标签的分类结果，从K=0到K=4，K越小则表明预测的时间越小（即用一个离散的度来衡量执行时间的大小，而不是具体的时间单位），这里K取4是实验得到的最佳取值范围。我们使用了*softmax*函数在执行时间范围内生成适当的概率分布以及*dropout*正则化来防止过拟合。

我们训练我们的网络来最小化历史查询计划和它们相应的预测执行时间的交叉熵（多分类问题的标准损失），我们研究了卷积神经网络，但没有发现明显的性能改善。我们的模型在Join Order Benchmark（JOB）上达到了60.9%的准确率，尽管这个结果与其他的预测任务的结果不能相比，但经过MCTS的大量仿真来选择合适的连接顺序，将弥补网络模型的准确性（AlphaGo的准确率也只达到了50%，但任然表现出了良好的性能）。

#### MCTS for AlphaJoin

这一节我们首先介绍UCT算法，MCTS中的reward function，然后讨论MCTS是如何工作的。

**UCT Algorithm：**全称是"应用于树的上限置信区间算法（Upper Confidence Bounds applied to Trees, UCT ）"，这是一种博弈树搜索算法，用来解决选择树节点的问题。该算法采不仅拥有搜索模型的能力，还能挖掘更多以前从未尝试过的树节点，以减少陷入局部最优的可能性。公式如下：
$$
UCT(v_i,v)=\frac{Q(v_i)}{N(v_i)} + C\sqrt{\frac{logN(v)}{N(v_i)}}
$$

其中$v_i$是当前节点，$v$是它的父节点，$Q(v_i)$表示在当前节点获得优势的次数，$N(v_i)(N(v))$是当前节点访问的总次数，$C$是探索敏感度参数。

以AlphaGo为例，公式的第一部分称为exploitation，是当前节点获得优势的次数与搜索次数的比值，这决定了选择该节点后获胜的概率，第二部分称为exploration，若子节点访问次数$N(v_i)$越小则这部分的值越大，这让未被充分搜索的节点获得了更大的被搜索的机会。

**Reward Function：**与AlphaGo类似，在MCTS获得一个完整的连接顺序后，我们需要定义"获胜"情况。在查询优化问题中，我们的目标是生成一个具有最小执行时间的查询计划，因此，更小的执行时间意味着更高的获胜可能。我们设计了一个简单的奖励函数如下：
$$
R_j=\frac{K-k_j}{K} ,\quad k_j=0,1,2,3,4
$$
其中$K$是前面在OVN中定义的执行时间的度，$k_j$是尝试的连接顺序$j$所预测的执行时间的度，这个函数取值为1则该计划的执行时间表明预测的执行时间的度是最小值0，即效果最好。

**MCTS for Join Order Selection：** MCTS算法是一种应用蒙特卡洛方法进行树搜索的决策算法。MCTS通过模拟将树展开，在搜索空间直到做出决策，并将决策的最终结果反馈给树中的节点进行更新。经过大量的模拟后，每个节点的信息表示为正确决策的数量与该节点的模拟总数的比例。为了降低从根节点到叶子节点的搜索空间，我们将每个节点需要模拟的次数定义为$S_n=N_c*F_s$，其中$N_c$是当前节点下子节点的数量，$F_S$是控制搜索次数的因子。每个SQL查询的模拟次数是$\sum^J_1S_n$，其中$J$是本次查询的连接数。MCTS的过程会持续到搜索完所有的连接顺序或者达到最大模拟次数，每次模拟可以分为以下四个步骤：

1. **Selection-**用UCT算法来选择从根节点到叶子节点的连接顺序，一旦在遍历过程中遇到子节点，则进入expansion步骤。
2. **Expansion-**如果选择的当前节点没有达到终止条件，则创建一个该节点的子节点.
3. **Simulation-**用随机策略向下进行选择模拟直至达到终止节点。
4. **Back Propagation-**从新节点到根节点执行反向传播过程。在这个过程中，每个节点中存储的模拟总数增加1，如果新节点的仿真结果执行时间更短，这些节点也会增加更新reward。

这个过程见下图

![图3](/images/AlphaJoin%20%E8%BF%9E%E6%8E%A5%E9%A1%BA%E5%BA%8F%E9%80%89%E6%8B%A9%E5%99%A8/MCTS%E8%BF%87%E7%A8%8B.png)

**Preliminary Results for AlphaJoin 1.0：**我们首先探索了$F_s$对选择器性能的影响，然后通过一系列基准测试对比了AlphaJoin、PostgreSQL的查询优化器、NEO之间的性能，结果如下图

![图4](/images/AlphaJoin%20%E8%BF%9E%E6%8E%A5%E9%A1%BA%E5%BA%8F%E9%80%89%E6%8B%A9%E5%99%A8/AlphaJoin1.0%E6%80%A7%E8%83%BD.png)

图(a)说明了不同的因子$F_s$对优化执行时间和搜索时间的影响，我们列举了$F_s$从5到25的取值，发现AlphaJoin的优化执行时间不断减小，但搜索时间逐渐增加，因为$F_s$越大MCTS过程的则模拟次数越大。因此，进行一个综合权衡后我们认为最佳的$F_s$取值是15。

图(b)展示了三种优化器的性能差异，说明即使我们将搜索时间也考虑在内，AlphaJoin的性能任然高于另外两者。实验表明MCTS能有效地选择合适的连接顺序。

### AlphaJoin2.0

在上图中的图(c)中可以发现，并非所有测试样例AlphaJoin1.0的性能都比PostgreSQL的查询优化器要好，根据统计数据，由PostgreSQL的查询优化器在其各自的执行引擎上优化的查询中，仍然有大约48%的性能比AlphaJoin1.0的性能要好，我们认为是UCT算法的不确定性造成的这一结果。

一个有趣的发现是，尽管这两种方法在首选查询的数量上是相似的，但AlphaJoin 1.0与PostgreSQL的join枚举过程相比，实现了极大的性能改进，换句话说AlphaJoin在执行时间更长的查询计划上表现得更好，这是因为这些缓慢的查询有更多的空间进行优化。相比之下，对于快速查询的优化，AlphaJoin1.0并不适用。为了进一步提升AlphaJoin1.0的性能，我们提出了AlphaJoin2.0。

为了解决上述问题，我们训练另一种称为自适应决策网络( Adaptive Decision Network, ADN)在AlphaJoin 1.0和PostgreSQL优化器之间进行选择，如图2所示。ADN是从PostgreSQL优化器和AlphaJoin 1.0的历史执行时间的标记数据集学习的。ADN与OVN唯一的区别是输入和输出，输入是SQL-encoding，输出是一个二分类结果，它指示应该选择执行哪个优化器。这个架构我们称之为AlphaJoin2.0。

### 初步结果

AlphaJoin2.0与PostgreSQL、NEO、AlphaJoin1.0的性能对比如图3中的图(d)所示，可以看到AlphaJoin2.0拥有更好的性能，PostgreSQL表现更好的测试用例比例从48%降低到了9%。

## 结论与接下来的工作

在本文中，我们介绍了AlphaJoin，这是第一个将MCTS用于查询优化来产生高效的表连接顺序的数据库优化器。AlphaJoin包含两个神经网络和MCTS，初步的结果表明它的性能由于NEO（目前最先进的方法）和PostgreSQL的优化器。

接下来，我们计划研究提高OVN和ADN两种网络预测精度以及进一步优化MCTS的搜索效率。此外，我们的方法在一组快速查询上的性能仍然不如PostgreSQL中的优化器，我们需要进一步研究，以进一步提高整体性能，并提出AlphaJoin 3.0

## REFERENCES

[1] Benoit Dageville et al. Automatic sql tuning in oracle 10g. In *VLDB*, pages 1098–1109, 2004.

[2] Surajit Chaudhuri et al. Self-tuning database systems: A decade of progress. In *VLDB*, pages 3–14, 2007.

[3] Dana Van Aken et al. Automatic dbms tuning through large-scale machine learning. In *SIGMOD*, 2017.

[4] Ji Zhang et al. An end-to-end automatic cloud database tuning system using deep reinforcement learning. In *SIGMOD ’19*, page 415–432, 2019.

[5] Guoliang Li et al. Qtune: A query-aware database tuning system with deep reinforcement learning. *Proc.* *VLDB Endow.*, 12(12):2118–2130, 2019.

[6] Ryan Marcus et al. Deep reinforcement learning for join order enumeration. aiDM’18. ACM, 2018.

[7] Ryan Marcus. Towards a hands-free query optimizer through deep learning. In *CIDR*, pages 1–8, 2019.

[8] Ryan Marcus et al. Neo: A learned query optimizer. *Proc. VLDB Endow.*, 12(11):1705–1718, 2019.

[9] R´emi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In *Computers* *and Games*, volume 4630, pages 72–83. Springer, 2006.

[10] David Silver et al. Mastering the game of go with deep neural networks and tree search. *Nature*, 2016.

[11] Jennifer Ortiz et al. Learning state representations for query optimization with deep reinforcement learning. In *DEEM’18*. ACM, 2018.

[12] Nitish Srivastava et al. Dropout: A simple way to prevent neural networks from overfifitting. *J. Mach.* Learn. Res.*, 15(1):1929-1958, January 2014.

[13] Ian Goodfellow et al. *Deep Learning*. MIT Press, 2016.

[14] Sylvain Gelly et al. Exploration exploitation in go: Uct for monte-carlo go. In *NIPS Workshop OTEE*, 2006.

[15] Viktor Leis et al. How good are query optimizers, really? *Proc. VLDB Endow.*, 9(3):204–215, 2015.

[16] Anji Liu. Watch the unobserved: A simple approach to parallelizing monte carlo tree search, 2018.

