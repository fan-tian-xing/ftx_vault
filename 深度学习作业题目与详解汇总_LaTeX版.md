# 深度学习作业题目与详解汇总

---

# 第一次作业

## 一、香农熵

### 题目

在多类别分类问题中，某模型对 $K=4$ 个类别给出了以下概率预测分布：


$$
p=(0.5,0.25,0.125,0.125)^T.
$$


请计算该分布 $P$ 的香农熵 $H(P)$，使用 $\log_2$ 作为底，单位为比特。

### 解答

香农熵定义为：


$$
H(P)=-\sum_i p_i\log_2 p_i.
$$


代入题目中的概率：


$$
H(P)=-(0.5\log_2 0.5+0.25\log_2 0.25+0.125\log_2 0.125+0.125\log_2 0.125).
$$


逐项计算：


$$
\log_2 0.5=-1,
$$



$$
\log_2 0.25=-2,
$$


$$
\log_2 0.125=-3.
$$


因此：


$$
0.5\log_2 0.5=0.5(-1)=-0.5,
$$



$$
0.25\log_2 0.25=0.25(-2)=-0.5,
$$



$$
0.125\log_2 0.125=0.125(-3)=-0.375.
$$


由于最后两个概率都为 $0.125$，所以：


$$
H(P)=-[-0.5-0.5-0.375-0.375].
$$



$$
H(P)=1.75.
$$


所以该概率分布的香农熵为：


$$
\boxed{H(P)=1.75\text{ bits}}.
$$


---

## 二、多元高斯分布的 KL 散度

### 题目

设 $P$ 和 $Q$ 是 $n$ 维空间中的两个多元高斯分布：


$$
P\sim N(\mu_1,\Sigma_1),\qquad Q\sim N(\mu_2,\Sigma_2).
$$


1. 写出 $D_{KL}(P\|Q)$ 的一般解析表达式。
2. 假设 $n=2$，$\mu_1=\mu_2=0$，$\Sigma_1=I$，$\Sigma_2=4I$，计算 $D_{KL}(P\|Q)$ 的值。

### 解答

两个多元高斯分布之间的 KL 散度为：


$$
D_{KL}(P\|Q)
=\frac{1}{2}
\left[
\operatorname{tr}(\Sigma_2^{-1}\Sigma_1)
+(\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)
-n
+\ln\frac{\det\Sigma_2}{\det\Sigma_1}
\right].
$$


题目中：


$$
n=2,\qquad \mu_1=\mu_2=0,\qquad \Sigma_1=I,\qquad \Sigma_2=4I.
$$


先计算 $\Sigma_2^{-1}$：


$$
\Sigma_2^{-1}=(4I)^{-1}=\frac{1}{4}I.
$$


因此：


$$
\Sigma_2^{-1}\Sigma_1=\frac{1}{4}I\cdot I=\frac{1}{4}I.
$$


因为这里是二维，所以：


$$
\operatorname{tr}\left(\frac{1}{4}I\right)
=\frac{1}{4}+\frac{1}{4}
=\frac{1}{2}.
$$


均值相同，所以均值差项为：


$$
(\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)=0.
$$


再计算行列式：


$$
\det(\Sigma_1)=\det(I)=1.
$$



$$
\det(\Sigma_2)=\det(4I).
$$


因为是二维矩阵：


$$
\det(4I)=4^2=16.
$$


所以：


$$
\ln\frac{\det\Sigma_2}{\det\Sigma_1}
=\ln\frac{16}{1}
=\ln 16.
$$


代入 KL 散度公式：


$$
D_{KL}(P\|Q)
=\frac{1}{2}\left[
\frac{1}{2}+0-2+\ln16
\right].
$$



$$
D_{KL}(P\|Q)
=\frac{1}{2}\left[
\ln16-\frac{3}{2}
\right].
$$


由于：


$$
\frac{1}{2}\ln16=\ln4,
$$


所以：


$$
D_{KL}(P\|Q)=\ln4-\frac{3}{4}.
$$


数值上：


$$
\ln4\approx1.3863,
$$


因此：


$$
D_{KL}(P\|Q)\approx1.3863-0.75=0.6363.
$$


最终答案为：


$$
\boxed{D_{KL}(P\|Q)=\ln4-\frac{3}{4}\approx0.6363}.
$$


---

## 三、微分熵在变量变换下的关系

### 题目

设 $x$ 和 $y$ 是 $\mathbb{R}^n$ 上的连续随机向量，其概率密度函数分别为 $p_x(x)$ 和 $p_y(y)$。已知：


$$
y=f(x)
$$


是一个可微分且雅可比矩阵 $J_f(x)$ 非奇异的变换。

1. 证明随机变量 $y$ 的微分熵 $H(y)$ 与 $x$ 的微分熵 $H(x)$ 之间的关系满足：


$$
H(y)=H(x)-E_{p_y}[\ln|\det J_{f^{-1}}(y)|].
$$


2. 假设变换是线性的：


$$
y=Wx,
$$


其中 $x$ 和 $y$ 均为 $n$ 维向量，$W$ 是一个 $n\times n$ 的可逆矩阵。矩阵 $W$ 需要满足什么条件，才能确保变换后的微分熵 $H(y)$ 小于原始熵 $H(x)$？

### 解答

连续随机变量的微分熵定义为：


$$
H(x)=-\int p_x(x)\ln p_x(x)\,dx.
$$


因为 $y=f(x)$ 可逆且可微，所以密度变换公式为：


$$
p_y(y)=p_x(f^{-1}(y))\left|\det J_{f^{-1}}(y)\right|.
$$


对两边取对数：


$$
\ln p_y(y)
=\ln p_x(f^{-1}(y))
+\ln\left|\det J_{f^{-1}}(y)\right|.
$$


根据微分熵定义：


$$
H(y)=-E_{p_y}[\ln p_y(y)].
$$


代入上式：


$$
H(y)
=-E_{p_y}
\left[
\ln p_x(f^{-1}(y))
+\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$


拆成两项：


$$
H(y)
=-E_{p_y}[\ln p_x(f^{-1}(y))]
-E_{p_y}\left[
\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$


由于 $x=f^{-1}(y)$，当 $y\sim p_y$ 时，对应的 $x\sim p_x$，所以：


$$
-E_{p_y}[\ln p_x(f^{-1}(y))]
=-E_{p_x}[\ln p_x(x)]
=H(x).
$$


因此：


$$
H(y)=H(x)-E_{p_y}\left[
\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$


即：


$$
\boxed{
H(y)=H(x)-E_{p_y}[\ln|\det J_{f^{-1}}(y)|]
}.
$$


也可以写成等价形式。因为：


$$
J_{f^{-1}}(y)=J_f(x)^{-1},
$$


所以：


$$
\det J_{f^{-1}}(y)=\frac{1}{\det J_f(x)}.
$$


于是：


$$
\ln|\det J_{f^{-1}}(y)|
=-\ln|\det J_f(x)|.
$$


因此：


$$
H(y)=H(x)+E_{p_x}[\ln|\det J_f(x)|].
$$


当 $y=Wx$ 时：


$$
J_f(x)=W.
$$


所以：


$$
H(y)=H(x)+\ln|\det W|.
$$


若要求：


$$
H(y)<H(x),
$$


则必须有：


$$
H(x)+\ln|\det W|<H(x).
$$


即：


$$
\ln|\det W|<0.
$$


因此：


$$
|\det W|<1.
$$


又因为 $W$ 可逆，所以：


$$
\det W\neq0.
$$


最终条件为：


$$
\boxed{0<|\det W|<1}.
$$


---

## 四、Softmax 与交叉熵求导

### 题目

在多分类问题中，Softmax 函数的预测概率为：


$$
a_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
$$


损失函数使用交叉熵损失：


$$
L=-\sum_{k=1}^K y_k\ln a_k,
$$


其中 $y_k$ 是真实标签的 One-Hot 编码。请推导损失函数 $L$ 对 Softmax 输入 $z_i$ 的导数：


$$
\frac{\partial L}{\partial z_i}.
$$


### 解答

Softmax 输出为：


$$
a_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$


先求 Softmax 的导数。对 $z_i$ 求导，有两种情况。

当 $k=i$ 时：


$$
\frac{\partial a_i}{\partial z_i}
=a_i(1-a_i).
$$


当 $k\neq i$ 时：


$$
\frac{\partial a_k}{\partial z_i}
=-a_ka_i.
$$


这两种情况可以统一写成：


$$
\frac{\partial a_k}{\partial z_i}
=a_k(\delta_{ki}-a_i),
$$


其中 $\delta_{ki}$ 是 Kronecker delta：


$$
\delta_{ki}=
\begin{cases}
1, & k=i,\\
0, & k\neq i.
\end{cases}
$$


交叉熵损失为：


$$
L=-\sum_k y_k\ln a_k.
$$


先对 $a_k$ 求导：


$$
\frac{\partial L}{\partial a_k}
=-\frac{y_k}{a_k}.
$$


根据链式法则：


$$
\frac{\partial L}{\partial z_i}
=\sum_k
\frac{\partial L}{\partial a_k}
\frac{\partial a_k}{\partial z_i}.
$$


代入：


$$
\frac{\partial L}{\partial z_i}
=\sum_k
\left(-\frac{y_k}{a_k}\right)
a_k(\delta_{ki}-a_i).
$$


约去 $a_k$：


$$
\frac{\partial L}{\partial z_i}
=-\sum_k y_k(\delta_{ki}-a_i).
$$


展开：


$$
\frac{\partial L}{\partial z_i}
=-\sum_k y_k\delta_{ki}
+\sum_k y_ka_i.
$$


第一项：


$$
\sum_k y_k\delta_{ki}=y_i.
$$


第二项中 $a_i$ 与求和指标 $k$ 无关：


$$
\sum_k y_ka_i
=a_i\sum_k y_k.
$$


因为 $y$ 是 One-Hot 标签：


$$
\sum_k y_k=1.
$$


所以：


$$
\sum_k y_ka_i=a_i.
$$


因此：


$$
\frac{\partial L}{\partial z_i}
=-y_i+a_i.
$$


最终得到：


$$
\boxed{
\frac{\partial L}{\partial z_i}=a_i-y_i
}.
$$


向量形式为：


$$
\boxed{
\nabla_z L=a-y
}.
$$


---

# 第二次作业

## 一、多输出线性模型的 MSE 梯度与 Hessian

### 题目

考虑一个多输出的线性模型：


$$
f(V;x)=Vx\in\mathbb{R}^C,
$$


其中：


$$
V=(v_1^T;\cdots;v_C^T)\in\mathbb{R}^{C\times d},
$$


$v_i\in\mathbb{R}^d$ 是与第 $i$ 个输出相关的权重向量。

最小化均方误差损失函数：


$$
\ell_{MSE}(V)=\frac{1}{N}\sum_{n=1}^N \|Vx_n-Y_n\|_2^2.
$$


其中 $Y_n\in\mathbb{R}^C$ 是目标输出向量。计算损失函数关于 $V$ 的梯度，即：


$$
\frac{\partial \ell}{\partial v_i},
$$


以及 Hessian 矩阵：


$$
\frac{\partial^2 \ell}{\partial v_i\partial v_j}.
$$


### 解答

对第 $n$ 个样本，模型输出为：


$$
\hat{Y}_n=Vx_n.
$$


第 $i$ 个输出分量为：


$$
\hat{Y}_{n,i}=v_i^T x_n.
$$


损失函数可以按输出维度展开：


$$
\ell(V)
=\frac{1}{N}\sum_{n=1}^N
\sum_{i=1}^C
(v_i^T x_n-Y_{n,i})^2.
$$


现在对某一个 $v_i$ 求梯度。只有第 $i$ 个输出分量对应的项与 $v_i$ 有关，因此：


$$
\frac{\partial \ell}{\partial v_i}
=\frac{1}{N}\sum_{n=1}^N
\frac{\partial}{\partial v_i}
(v_i^T x_n-Y_{n,i})^2.
$$


令：


$$
e_{n,i}=v_i^T x_n-Y_{n,i}.
$$


则：


$$
\frac{\partial}{\partial v_i}e_{n,i}^2
=2e_{n,i}\frac{\partial e_{n,i}}{\partial v_i}.
$$


而：


$$
\frac{\partial e_{n,i}}{\partial v_i}
=\frac{\partial}{\partial v_i}(v_i^T x_n-Y_{n,i})
=x_n.
$$


因此：


$$
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n.
$$


即：


$$
\boxed{
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n
}.
$$


把所有输出分量合在一起，可以得到矩阵形式。记：


$$
e_n=Vx_n-Y_n.
$$


其中 $e_n\in\mathbb{R}^C$，则：


$$
\boxed{
\nabla_V\ell
=\frac{2}{N}\sum_{n=1}^N
(Vx_n-Y_n)x_n^T
}.
$$


接下来计算 Hessian。

先看 $i=j$ 的情况：


$$
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n.
$$


继续对 $v_i$ 求导：


$$
\frac{\partial^2 \ell}{\partial v_i^2}
=\frac{2}{N}\sum_{n=1}^N
x_nx_n^T.
$$


因此：


$$
\boxed{
\frac{\partial^2 \ell}{\partial v_i^2}
=\frac{2}{N}\sum_{n=1}^N x_nx_n^T
}.
$$


当 $i\neq j$ 时，第 $i$ 个输出误差：


$$
v_i^Tx_n-Y_{n,i}
$$


不含 $v_j$，所以：


$$
\boxed{
\frac{\partial^2 \ell}{\partial v_i\partial v_j}=0,\qquad i\neq j
}.
$$


因此 Hessian 是一个按输出维度分块的块对角矩阵：


$$
\boxed{
H=I_C\otimes
\left(
\frac{2}{N}\sum_{n=1}^N x_nx_n^T
\right)
}.
$$


其中 $\otimes$ 表示 Kronecker 积。

---

## 二、双月亮数据与带 $L_2$ 正则的逻辑回归

### 题目

使用 Python `sklearn.datasets.make_moons` 函数生成一个“双月亮”数据集，然后应用带有 $L_2$ 正则化的线性分类器，即逻辑回归。

具体要求：

1. 选择一个合适阶数的多项式基函数 $\phi(x)$，将原始二维特征 $x$ 映射到高维特征空间 $\phi(x)$。
2. 在映射后的高维特征空间上，训练两个逻辑回归模型：模型 A 无正则化或 $\lambda$ 极小；模型 B 使用强 $L_2$ 正则化，即 $\lambda$ 较大。
3. 在原始二维特征空间上，可视化两个模型的决策边界。
4. 针对带有 $L_2$ 正则化的逻辑回归模型，写出其损失函数，并推导最优解 $\hat{w}$ 满足的条件，即梯度为零的方程。

### 解答

双月亮数据在原始二维空间中通常不是线性可分的，因此直接使用线性分类器很难得到弯曲的分类边界。解决方法是先进行多项式特征映射：


$$
x=(x_1,x_2)^T
\quad\longmapsto\quad
\phi(x).
$$


例如二阶多项式映射可以写为：


$$
\phi(x_1,x_2)
=
(1,x_1,x_2,x_1^2,x_1x_2,x_2^2)^T.
$$


如果需要更复杂的非线性边界，也可以使用三阶或五阶多项式特征。

逻辑回归模型为：


$$
\hat{y}_n
=\sigma(w^T\phi(x_n)+b),
$$


其中：


$$
\sigma(t)=\frac{1}{1+e^{-t}}.
$$


令：


$$
z_n=w^T\phi(x_n)+b.
$$


则：


$$
\hat{y}_n=\sigma(z_n).
$$


带 $L_2$ 正则化的二分类交叉熵损失为：


$$
J(w,b)
=
\frac{1}{N}\sum_{n=1}^N
\left[
-y_n\ln \hat{y}_n
-(1-y_n)\ln(1-\hat{y}_n)
\right]
+\frac{\lambda}{2}\|w\|_2^2.
$$


通常不对偏置 $b$ 做正则化。

下面推导梯度。对逻辑回归而言：


$$
\frac{\partial J}{\partial z_n}
=\hat{y}_n-y_n.
$$


又因为：


$$
z_n=w^T\phi(x_n)+b,
$$


所以：


$$
\frac{\partial z_n}{\partial w}=\phi(x_n),
\qquad
\frac{\partial z_n}{\partial b}=1.
$$


因此：


$$
\frac{\partial J}{\partial w}
=
\frac{1}{N}\sum_{n=1}^N
(\hat{y}_n-y_n)\phi(x_n)
+\lambda w.
$$


矩阵形式下，令 $\Phi$ 为特征矩阵，每一行是 $\phi(x_n)^T$，令 $\hat{y}$ 和 $y$ 为 $N$ 维列向量，则：


$$
\boxed{
\nabla_w J
=
\frac{1}{N}\Phi^T(\hat{y}-y)+\lambda w
}.
$$


对偏置：


$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac{1}{N}\sum_{n=1}^N(\hat{y}_n-y_n)
}.
$$


最优解 $(\hat{w},\hat{b})$ 满足梯度为零：


$$
\boxed{
\frac{1}{N}\Phi^T(\hat{y}-y)+\lambda \hat{w}=0
}.
$$



$$
\boxed{
\frac{1}{N}\mathbf{1}^T(\hat{y}-y)=0
}.
$$


需要注意的是，逻辑回归带 sigmoid 非线性，一般没有像普通最小二乘那样的显式闭式解。上面的方程是最优解必须满足的一阶最优条件，通常用梯度下降、牛顿法、拟牛顿法或 sklearn 内部优化器数值求解。

模型 A 和模型 B 的区别：

模型 A：无正则化或 $\lambda$ 极小，允许更大的权重，决策边界通常更弯曲，拟合能力更强，但更容易过拟合。

模型 B：$\lambda$ 较大，权重被明显惩罚，决策边界通常更平滑、更简单，泛化能力可能更好，但过强正则也可能欠拟合。

代码与决策边界截图位置：

```text








```

---

## 三、两层隐藏层神经网络的反向传播

### 题目

使用 Python `sklearn.datasets.make_moons` 函数生成一个“双月亮”数据集。搭建含有 2 个隐藏层的神经网络，第一个隐藏层有 10 个神经元，第二个隐藏层有 20 个神经元，每层神经元使用 sigmoid 激活函数。损失函数使用二分类交叉熵：


$$
\ell(y,\hat{y})=-y\log(\hat{y})-(1-y)\log(1-\hat{y}).
$$


1. 利用反向传播算法，写出每层权重和偏置的梯度。
2. 当 $W^{(l)}$ 和 $b^{(l)}$ 都为 0 时，计算其对应的梯度值。

### 解答

设网络输入为：


$$
a^{(0)}=x.
$$


第一层隐藏层：


$$
z^{(1)}=W^{(1)}a^{(0)}+b^{(1)},
$$



$$
a^{(1)}=\sigma(z^{(1)}).
$$


第二层隐藏层：


$$
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)},
$$



$$
a^{(2)}=\sigma(z^{(2)}).
$$


输出层：


$$
z^{(3)}=W^{(3)}a^{(2)}+b^{(3)},
$$



$$
\hat{y}=a^{(3)}=\sigma(z^{(3)}).
$$


二分类交叉熵损失为：


$$
\ell
=-y\ln\hat{y}-(1-y)\ln(1-\hat{y}).
$$


对于 sigmoid 输出层加二分类交叉熵，有经典结果：


$$
\delta^{(3)}
=\frac{\partial \ell}{\partial z^{(3)}}
=\hat{y}-y.
$$


因此输出层梯度为：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(3)}}
=\delta^{(3)}(a^{(2)})^T
}.
$$



$$
\boxed{
\frac{\partial \ell}{\partial b^{(3)}}
=\delta^{(3)}
}.
$$


第二层隐藏层误差为：


$$
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
\sigma'(z^{(2)}).
$$


因为 sigmoid 的导数为：


$$
\sigma'(z)=\sigma(z)(1-\sigma(z)),
$$


所以：


$$
\sigma'(z^{(2)})
=a^{(2)}\odot(1-a^{(2)}).
$$


因此：


$$
\boxed{
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
a^{(2)}\odot(1-a^{(2)})
}.
$$


第二层权重和偏置梯度：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(2)}}
=\delta^{(2)}(a^{(1)})^T
}.
$$



$$
\boxed{
\frac{\partial \ell}{\partial b^{(2)}}
=\delta^{(2)}
}.
$$


第一层隐藏层误差为：


$$
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
\sigma'(z^{(1)}).
$$


即：


$$
\boxed{
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
a^{(1)}\odot(1-a^{(1)})
}.
$$


第一层权重和偏置梯度：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(1)}}
=\delta^{(1)}(a^{(0)})^T
}.
$$



$$
\boxed{
\frac{\partial \ell}{\partial b^{(1)}}
=\delta^{(1)}
}.
$$


如果是 mini-batch 或全数据集训练，则每层梯度通常对样本求平均。例如：


$$
\frac{\partial L}{\partial W^{(l)}}
=
\frac{1}{N}\sum_{n=1}^N
\delta_n^{(l)}(a_n^{(l-1)})^T.
$$


下面讨论所有 $W^{(l)}$ 和 $b^{(l)}$ 都为 0 的情况。

因为：


$$
z^{(1)}=0,
$$


所以：


$$
a^{(1)}=\sigma(0)=0.5.
$$


这里 $a^{(1)}$ 是 10 维向量，所有分量都为 $0.5$。

同理：


$$
z^{(2)}=0,\qquad a^{(2)}=\sigma(0)=0.5.
$$


这里 $a^{(2)}$ 是 20 维向量，所有分量都为 $0.5$。

输出层：


$$
z^{(3)}=0,
$$


所以：


$$
\hat{y}=a^{(3)}=\sigma(0)=0.5.
$$


输出层误差为：


$$
\delta^{(3)}=\hat{y}-y=0.5-y.
$$


因此输出层偏置梯度为：


$$
\boxed{
\frac{\partial \ell}{\partial b^{(3)}}=0.5-y
}.
$$


输出层权重梯度为：


$$
\frac{\partial \ell}{\partial W^{(3)}}
=\delta^{(3)}(a^{(2)})^T.
$$


由于 $a^{(2)}$ 的每个分量都是 $0.5$，所以：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(3)}}
=(0.5-y)(0.5,\ldots,0.5)
}.
$$


接着看第二层隐藏层：


$$
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
a^{(2)}\odot(1-a^{(2)}).
$$


因为：


$$
W^{(3)}=0,
$$


所以：


$$
(W^{(3)})^T\delta^{(3)}=0.
$$


因此：


$$
\boxed{\delta^{(2)}=0}.
$$


从而：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(2)}}=0
},
\qquad
\boxed{
\frac{\partial \ell}{\partial b^{(2)}}=0
}.
$$


同理：


$$
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
a^{(1)}\odot(1-a^{(1)}).
$$


由于：


$$
\delta^{(2)}=0,
$$


所以：


$$
\boxed{\delta^{(1)}=0}.
$$


因此：


$$
\boxed{
\frac{\partial \ell}{\partial W^{(1)}}=0
},
\qquad
\boxed{
\frac{\partial \ell}{\partial b^{(1)}}=0
}.
$$


结论：当所有权重和偏置都初始化为 0 时，只有输出层的梯度可能非零，隐藏层梯度为零。这说明全零初始化会导致隐藏层无法有效学习，因此实际训练神经网络时通常不能把所有权重都初始化为 0。

代码与运行结果截图位置：

```text








```

---

# 第三次作业

## 一、卷积神经网络

### 题目 1：卷积计算

计算如下矩阵卷积：


$$
\begin{pmatrix}
2 & 1 & 1\\
4 & 3 & 5\\
7 & 6 & 0
\end{pmatrix}
*
\begin{pmatrix}
1 & 1\\
-1 & 1
\end{pmatrix}.
$$


### 解答

在深度学习课程中，通常所说的卷积实际使用的是 cross-correlation，即卷积核不旋转，直接滑动相乘求和。

输入矩阵大小为 $3\times3$，卷积核大小为 $2\times2$，不补零、步长为 1，因此输出矩阵大小为：


$$
(3-2+1)\times(3-2+1)=2\times2.
$$


卷积核为：


$$
K=
\begin{pmatrix}
1 & 1\\
-1 & 1
\end{pmatrix}.
$$


左上角窗口为：


$$
\begin{pmatrix}
2 & 1\\
4 & 3
\end{pmatrix}.
$$


对应输出：


$$
2\cdot1+1\cdot1+4\cdot(-1)+3\cdot1
=2+1-4+3=2.
$$


右上角窗口为：


$$
\begin{pmatrix}
1 & 1\\
3 & 5
\end{pmatrix}.
$$


对应输出：


$$
1\cdot1+1\cdot1+3\cdot(-1)+5\cdot1
=1+1-3+5=4.
$$


左下角窗口为：


$$
\begin{pmatrix}
4 & 3\\
7 & 6
\end{pmatrix}.
$$


对应输出：


$$
4\cdot1+3\cdot1+7\cdot(-1)+6\cdot1
=4+3-7+6=6.
$$


右下角窗口为：


$$
\begin{pmatrix}
3 & 5\\
6 & 0
\end{pmatrix}.
$$


对应输出：


$$
3\cdot1+5\cdot1+6\cdot(-1)+0\cdot1
=3+5-6+0=2.
$$


所以输出为：


$$
\boxed{
\begin{pmatrix}
2 & 4\\
6 & 2
\end{pmatrix}
}.
$$


如果严格按照数学卷积定义，需要先将卷积核旋转 $180^\circ$。但 CNN 中一般默认使用不旋转的 cross-correlation。

---

### 题目 2：卷积输出尺寸公式

考虑矩阵卷积。如果输入图片大小是 $W_1\times H_1$，滤波器大小是 $F\times F$，两端补零 $P$，步长为 $S$，卷积之后的长度为 $W_2\times H_2$，求 $W_2$ 和 $H_2$。

### 解答

先看宽度方向。

输入宽度为 $W_1$，左右两边各补零 $P$，所以补零后的有效宽度为：


$$
W_1+2P.
$$


卷积核宽度为 $F$，因此第一个卷积窗口占据 $F$ 个位置。若步长为 $S$，窗口每次移动 $S$ 个单位。

最后一个合法窗口的起始位置不能超过：


$$
W_1+2P-F.
$$


从位置 0 开始，每次移动 $S$，因此一共有：


$$
\left\lfloor \frac{W_1-F+2P}{S}\right\rfloor+1
$$


个位置。

所以：


$$
\boxed{
W_2=
\left\lfloor
\frac{W_1-F+2P}{S}
\right\rfloor+1
}.
$$


同理，高度方向：


$$
\boxed{
H_2=
\left\lfloor
\frac{H_1-F+2P}{S}
\right\rfloor+1
}.
$$


如果题目保证整除，也可以不写 floor。

---

### 题目 3：卷积的平移等变性证明

证明卷积算子保持平移等变性：


$$
C(T_{a,b}\circ x)=T_{a,b}\circ C(x),
$$


其中：


$$
C(x)=x*w,
$$


$T_{a,b}$ 表示平移算子，且：


$$
(T_{a,b}\circ x)_{i,j}=x_{i-a,j-b}.
$$


### 解答

要证明两个矩阵相等，只需要证明它们在任意位置 $(i,j)$ 的元素相等。

先计算左边：


$$
\left(C(T_{a,b}\circ x)\right)_{i,j}.
$$


按照卷积定义：


$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
\sum_{u,v}
(T_{a,b}\circ x)_{i-u,j-v}w_{u,v}.
$$


由平移算子定义：


$$
(T_{a,b}\circ x)_{i-u,j-v}
=x_{i-u-a,j-v-b}.
$$


因此：


$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$


再看右边：


$$
(T_{a,b}\circ C(x))_{i,j}.
$$


由平移定义：


$$
(T_{a,b}\circ C(x))_{i,j}
=C(x)_{i-a,j-b}.
$$


而：


$$
C(x)_{i-a,j-b}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$


所以：


$$
(T_{a,b}\circ C(x))_{i,j}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$


这与左边完全相同，因此：


$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
(T_{a,b}\circ C(x))_{i,j}.
$$


由于对任意 $(i,j)$ 都成立，所以：


$$
\boxed{
C(T_{a,b}\circ x)=T_{a,b}\circ C(x)
}.
$$


这说明卷积是平移等变的：输入平移多少，输出也相应平移多少。

---

### 题目 4：残差神经网络与 Batch Normalization

简单介绍残差神经网络，并分析该网络为什么需要加入 Batch Normalization 层。

### 解答

残差神经网络 ResNet 的核心结构是残差连接，也称 shortcut connection 或 identity connection。

普通神经网络希望直接学习目标映射：


$$
H(x).
$$


ResNet 不直接让网络学习 $H(x)$，而是让网络学习残差部分：


$$
F(x)=H(x)-x.
$$


于是原来的目标映射可以写成：


$$
H(x)=F(x)+x.
$$


因此残差模块的输出为：


$$
\boxed{
y=F(x)+x
}.
$$


这样做的好处是，如果某些层暂时不需要学习复杂变换，网络只需要让 $F(x)\approx0$，整个模块就近似为恒等映射：


$$
y\approx x.
$$


这比让若干非线性层直接学习恒等映射更容易。

ResNet 的主要优点包括：

1. 缓解深层网络退化问题。
2. 梯度可以通过 shortcut 更容易向前层传播。
3. 使非常深的网络更容易优化。
4. 让网络学习残差变化，而不是完整映射，降低优化难度。

Batch Normalization 的作用是对每一层的激活进行标准化，使其均值和方差更稳定。对一个 mini-batch 中的激活 $x$，BN 通常计算：


$$
\mu_B=\frac{1}{m}\sum_{i=1}^m x_i,
$$



$$
\sigma_B^2=\frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2.
$$


然后标准化：


$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}.
$$


再通过可学习参数恢复表达能力：


$$
y_i=\gamma \hat{x}_i+\beta.
$$


在 ResNet 中加入 BN 的原因包括：

1. 稳定每层输入分布，使训练更平稳。
2. 缓解梯度消失或梯度爆炸。
3. 加快网络收敛速度。
4. 使残差分支 $F(x)$ 的数值尺度更加稳定，方便与 shortcut 分支 $x$ 相加。
5. 对深层网络尤其重要，因为网络越深，激活分布和梯度尺度越容易变得不稳定。

因此，ResNet 通常将卷积层、BN 层和激活函数组合使用，使残差分支更容易被优化。

---

## 二、循环神经网络反向传播

### 题目

考虑有两个输入 $X_1$ 和 $X_2$，一个输出 $Y$，两个隐藏状态 $h_1,h_2$ 的循环神经网络：


$$
h_1=\tanh(Wh_0+UX_1+b),
$$



$$
h_2=\tanh(Wh_1+UX_2+b),
$$



$$
Y=Vh_2+c.
$$


假设所有参数都是一维的，损失函数为：


$$
L=\frac{1}{2}(Y-Z)^2,
$$


其中 $Z$ 是真实标签。计算：


$$
\frac{\partial L}{\partial W},
\quad
\frac{\partial L}{\partial U},
\quad
\frac{\partial L}{\partial V},
\quad
\frac{\partial L}{\partial b},
\quad
\frac{\partial L}{\partial c}.
$$


### 解答

先定义中间变量：


$$
s_1=Wh_0+UX_1+b,
$$



$$
h_1=\tanh(s_1).
$$



$$
s_2=Wh_1+UX_2+b,
$$



$$
h_2=\tanh(s_2).
$$



$$
Y=Vh_2+c.
$$


损失函数：


$$
L=\frac{1}{2}(Y-Z)^2.
$$


令：


$$
e=Y-Z.
$$


则：


$$
\frac{\partial L}{\partial Y}=Y-Z=e.
$$


先求输出层参数 $V,c$。

因为：


$$
Y=Vh_2+c,
$$


所以：


$$
\frac{\partial Y}{\partial V}=h_2,
\qquad
\frac{\partial Y}{\partial c}=1.
$$


因此：


$$
\boxed{
\frac{\partial L}{\partial V}
=eh_2
}.
$$



$$
\boxed{
\frac{\partial L}{\partial c}
=e
}.
$$


接下来把误差传到第二个时间步。

因为：


$$
\frac{\partial L}{\partial h_2}
=
\frac{\partial L}{\partial Y}
\frac{\partial Y}{\partial h_2}
=eV.
$$


又因为：


$$
h_2=\tanh(s_2),
$$


且：


$$
\frac{d}{ds}\tanh(s)=1-\tanh^2(s),
$$


所以：


$$
\frac{\partial h_2}{\partial s_2}
=1-h_2^2.
$$


令：


$$
g_2=\frac{\partial L}{\partial s_2}.
$$


则：


$$
g_2=
\frac{\partial L}{\partial h_2}
\frac{\partial h_2}{\partial s_2}
=eV(1-h_2^2).
$$


即：


$$
\boxed{
g_2=eV(1-h_2^2)
}.
$$


再把误差传到第一个时间步。

由于：


$$
s_2=Wh_1+UX_2+b,
$$


所以：


$$
\frac{\partial s_2}{\partial h_1}=W.
$$


因此：


$$
\frac{\partial L}{\partial h_1}
=
\frac{\partial L}{\partial s_2}
\frac{\partial s_2}{\partial h_1}
=g_2W.
$$


又因为：


$$
h_1=\tanh(s_1),
$$


所以：


$$
\frac{\partial h_1}{\partial s_1}
=1-h_1^2.
$$


令：


$$
g_1=\frac{\partial L}{\partial s_1}.
$$


则：


$$
g_1
=
\frac{\partial L}{\partial h_1}
\frac{\partial h_1}{\partial s_1}
=g_2W(1-h_1^2).
$$


即：


$$
\boxed{
g_1=g_2W(1-h_1^2)
}.
$$


现在计算共享参数 $W,U,b$ 的梯度。由于 $W,U,b$ 在两个时间步中共享，所以总梯度等于两个时间步贡献之和。

对 $W$：


$$
s_1=Wh_0+UX_1+b
\quad\Rightarrow\quad
\frac{\partial s_1}{\partial W}=h_0.
$$



$$
s_2=Wh_1+UX_2+b
\quad\Rightarrow\quad
\frac{\partial s_2}{\partial W}=h_1.
$$


所以：


$$
\boxed{
\frac{\partial L}{\partial W}
=g_1h_0+g_2h_1
}.
$$


对 $U$：


$$
\frac{\partial s_1}{\partial U}=X_1,
\qquad
\frac{\partial s_2}{\partial U}=X_2.
$$


所以：


$$
\boxed{
\frac{\partial L}{\partial U}
=g_1X_1+g_2X_2
}.
$$


对 $b$：


$$
\frac{\partial s_1}{\partial b}=1,
\qquad
\frac{\partial s_2}{\partial b}=1.
$$


所以：


$$
\boxed{
\frac{\partial L}{\partial b}
=g_1+g_2
}.
$$


综上：


$$
\boxed{
\frac{\partial L}{\partial V}=eh_2
},
\qquad
\boxed{
\frac{\partial L}{\partial c}=e
}.
$$



$$
\boxed{
g_2=eV(1-h_2^2)
},
\qquad
\boxed{
g_1=g_2W(1-h_1^2)
}.
$$



$$
\boxed{
\frac{\partial L}{\partial W}=g_2h_1+g_1h_0
}.
$$



$$
\boxed{
\frac{\partial L}{\partial U}=g_2X_2+g_1X_1
}.
$$



$$
\boxed{
\frac{\partial L}{\partial b}=g_2+g_1
}.
$$


---

## 三、行 Softmax 与置换矩阵

### 题目

设 $P\in\mathbb{R}^{N\times N}$ 为置换矩阵，$A\in\mathbb{R}^{N\times N}$ 为任意矩阵。记 $\operatorname{Softmax}(A)$ 表示对矩阵 $A$ 的每一行分别施加 softmax 操作。证明：


$$
\operatorname{Softmax}(PAP^T)
=P\operatorname{Softmax}(A)P^T.
$$


### 解答

置换矩阵 $P$ 的作用是重新排列向量或矩阵的行与列。

假设 $P$ 对应的置换为 $\pi$。则矩阵：


$$
B=PAP^T
$$


的第 $(i,j)$ 个元素为：


$$
B_{ij}=A_{\pi(i),\pi(j)}.
$$


对 $B$ 的第 $i$ 行做 softmax：


$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{B_{ij}}}{\sum_{k=1}^N e^{B_{ik}}}.
$$


代入 $B_{ij}=A_{\pi(i),\pi(j)}$：


$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{A_{\pi(i),\pi(j)}}}
{\sum_{k=1}^N e^{A_{\pi(i),\pi(k)}}}.
$$


因为 $\pi(k)$ 只是对 $1,\ldots,N$ 的重新排列，所以：


$$
\sum_{k=1}^N e^{A_{\pi(i),\pi(k)}}
=
\sum_{m=1}^N e^{A_{\pi(i),m}}.
$$


因此：


$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{A_{\pi(i),\pi(j)}}}
{\sum_{m=1}^N e^{A_{\pi(i),m}}}.
$$


这正是：


$$
\operatorname{Softmax}(A)_{\pi(i),\pi(j)}.
$$


另一方面，矩阵：


$$
P\operatorname{Softmax}(A)P^T
$$


的第 $(i,j)$ 个元素也是：


$$
\operatorname{Softmax}(A)_{\pi(i),\pi(j)}.
$$


所以：


$$
\operatorname{Softmax}(PAP^T)_{ij}
=
\left(P\operatorname{Softmax}(A)P^T\right)_{ij}.
$$


由于对任意 $i,j$ 都成立，因此：


$$
\boxed{
\operatorname{Softmax}(PAP^T)
=P\operatorname{Softmax}(A)P^T
}.
$$


---

## 四、自注意力的置换等变性及其与 CNN 的关系

### 题目

设输入矩阵为 $X\in\mathbb{R}^{N\times D}$，其中每一行表示一个 token 的特征向量。定义单头自注意力模块为：


$$
\operatorname{Att}(X)
=
\operatorname{Softmax}
\left(
\frac{XW_Q(XW_K)^T}{\sqrt{d_k}}
\right)
XW_V,
$$


其中：


$$
W_Q\in\mathbb{R}^{D\times d_k},
\qquad
W_K\in\mathbb{R}^{D\times d_k},
\qquad
W_V\in\mathbb{R}^{D\times d_v}.
$$


1. 证明对任意置换矩阵 $P$，有：


$$
\operatorname{Att}(PX)=P\operatorname{Att}(X).
$$


因此，自注意力模块对输入 token 的置换是等变的。

2. 分析自注意力模型和卷积神经网络之间的关系。

### 解答

先定义：


$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$


于是：


$$
\operatorname{Att}(X)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
$$


现在考虑输入变为 $PX$。新的 query、key、value 分别为：


$$
Q'=(PX)W_Q=P(XW_Q)=PQ.
$$



$$
K'=(PX)W_K=P(XW_K)=PK.
$$



$$
V'=(PX)W_V=P(XW_V)=PV.
$$


因此新的注意力分数矩阵为：


$$
Q'(K')^T
=(PQ)(PK)^T.
$$


由于：


$$
(PK)^T=K^TP^T,
$$


所以：


$$
Q'(K')^T
=P QK^T P^T.
$$


因此：


$$
\frac{Q'(K')^T}{\sqrt{d_k}}
=
P
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
P^T.
$$


由上一题已经证明的行 softmax 与置换矩阵关系：


$$
\operatorname{Softmax}(PAP^T)
=P\operatorname{Softmax}(A)P^T.
$$


令：


$$
A=\frac{QK^T}{\sqrt{d_k}},
$$


可得：


$$
\operatorname{Softmax}
\left(
\frac{Q'(K')^T}{\sqrt{d_k}}
\right)
=
P
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
P^T.
$$


于是：


$$
\operatorname{Att}(PX)
=
\operatorname{Softmax}
\left(
\frac{Q'(K')^T}{\sqrt{d_k}}
\right)
V'.
$$


代入上式和 $V'=PV$：


$$
\operatorname{Att}(PX)
=
P
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
P^T
PV.
$$


因为：


$$
P^TP=I,
$$


所以：


$$
\operatorname{Att}(PX)
=
P
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V.
$$


而：


$$
\operatorname{Att}(X)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V.
$$


所以：


$$
\boxed{
\operatorname{Att}(PX)=P\operatorname{Att}(X)
}.
$$


这说明：如果输入 token 的顺序被同一个置换矩阵 $P$ 重排，那么输出 token 也会以相同方式重排。因此自注意力本身具有置换等变性。

下面分析自注意力与 CNN 的关系。

CNN 的典型形式可以理解为：


$$
y_i=\sum_{j\in\mathcal{N}(i)} w_{i-j}x_j.
$$


其中：

$\mathcal{N}(i)$ 是位置 $i$ 附近的局部邻域，$w_{i-j}$ 是卷积核权重。

CNN 的特点：

1. 权重是固定的，训练完成后与具体输入内容无关。
2. 聚合范围通常是局部邻域。
3. 卷积核在不同空间位置共享。
4. 具有平移等变性。

自注意力可以写成：


$$
y_i=\sum_{j=1}^N \alpha_{ij}v_j.
$$


其中：


$$
\alpha_{ij}
=
\frac{\exp(q_i^Tk_j/\sqrt{d_k})}
{\sum_m \exp(q_i^Tk_m/\sqrt{d_k})}.
$$


自注意力的特点：

1. 权重 $\alpha_{ij}$ 由输入内容动态计算。
2. 每个 token 可以和所有 token 建立联系，因此天然具有全局建模能力。
3. 权重不是固定卷积核，而是随样本变化。
4. 如果不加入位置编码，自注意力对 token 顺序没有内置位置感。

二者的共同点是：输出本质上都是对输入特征的加权求和。

区别是：CNN 使用固定的、局部的、位置共享的权重；自注意力使用动态的、内容相关的、通常全局的权重。

因此可以说，自注意力可以看作一种更灵活的动态卷积。它能够在一定条件下模拟卷积操作，但表达能力更依赖输入内容和注意力权重。

---

## 五、图拉普拉斯公式证明

### 题目

设：


$$
x=(x_1,x_2,\ldots,x_N)^T\in\mathbb{R}^N
$$


是定义在图节点上的图信号，其中 $x_i$ 表示节点 $i$ 上的信号值。证明：


$$
x^TLx
=
\frac{1}{2}
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}(x_i-x_j)^2.
$$


其中：


$$
L=D-A
$$


表示图拉普拉斯矩阵，$D$ 表示度矩阵，$A$ 表示连接矩阵。

### 解答

图拉普拉斯矩阵定义为：


$$
L=D-A.
$$


因此：


$$
x^TLx=x^T(D-A)x.
$$


展开：


$$
x^TLx=x^TDx-x^TAx.
$$


先看第一项。因为 $D$ 是度矩阵，是对角矩阵：


$$
D_{ii}=d_i.
$$


所以：


$$
x^TDx=\sum_{i=1}^N d_ix_i^2.
$$


对于无向图：


$$
d_i=\sum_{j=1}^N A_{ij}.
$$


因此：


$$
x^TDx
=
\sum_{i=1}^N
\left(
\sum_{j=1}^N A_{ij}
\right)x_i^2
=
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}x_i^2.
$$


再看第二项：


$$
x^TAx
=
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}x_ix_j.
$$


所以：


$$
x^TLx
=
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}x_i^2
-
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}x_ix_j.
$$


即：


$$
x^TLx
=
\sum_{i,j}A_{ij}x_i^2
-
\sum_{i,j}A_{ij}x_ix_j.
$$


现在展开右边要证明的式子：


$$
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i-x_j)^2.
$$


因为：


$$
(x_i-x_j)^2=x_i^2-2x_ix_j+x_j^2,
$$


所以：


$$
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i-x_j)^2
=
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i^2-2x_ix_j+x_j^2).
$$


展开：


$$
=
\frac{1}{2}
\sum_{i,j}
A_{ij}x_i^2
-
\sum_{i,j}
A_{ij}x_ix_j
+
\frac{1}{2}
\sum_{i,j}
A_{ij}x_j^2.
$$


对于无向图，邻接矩阵 $A$ 是对称的，即：


$$
A_{ij}=A_{ji}.
$$


因此：


$$
\sum_{i,j}A_{ij}x_j^2
=
\sum_{i,j}A_{ji}x_i^2
=
\sum_{i,j}A_{ij}x_i^2.
$$


所以第一项和第三项相加为：


$$
\frac{1}{2}
\sum_{i,j}
A_{ij}x_i^2
+
\frac{1}{2}
\sum_{i,j}
A_{ij}x_i^2
=
\sum_{i,j}
A_{ij}x_i^2.
$$


因此：


$$
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i-x_j)^2
=
\sum_{i,j}
A_{ij}x_i^2
-
\sum_{i,j}
A_{ij}x_ix_j.
$$


这正好等于：


$$
x^TLx.
$$


所以：


$$
\boxed{
x^TLx
=
\frac{1}{2}
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}(x_i-x_j)^2
}.
$$


这个公式说明，图拉普拉斯二次型衡量的是相邻节点信号值的差异。如果相连节点的信号越不相似，则 $(x_i-x_j)^2$ 越大，整体 $x^TLx$ 也越大。

---

# 图片题

## 图片题 1：DDPM 后验 $q(z_{t-1}\mid z_t,x)$

### 题目

已知 DDPM 前向扩散过程：


$$
q(z_t\mid z_{t-1})
=
N\left(
\sqrt{1-\beta_t}z_{t-1},
\beta_t I
\right),
$$


并且：


$$
q(z_t\mid x)
=
N\left(
\sqrt{\alpha_t}x,
(1-\alpha_t)I
\right),
$$


其中：


$$
\alpha_t=\prod_{s=1}^t(1-\beta_s).
$$


求：


$$
q(z_{t-1}\mid z_t,x).
$$


### 解答

我们希望求的是：


$$
q(z_{t-1}\mid z_t,x).
$$


由贝叶斯公式：


$$
q(z_{t-1}\mid z_t,x)
\propto
q(z_t\mid z_{t-1},x)q(z_{t-1}\mid x).
$$


由于前向扩散满足马尔可夫性质，给定 $z_{t-1}$ 后，$z_t$ 与 $x$ 条件独立，因此：


$$
q(z_t\mid z_{t-1},x)=q(z_t\mid z_{t-1}).
$$


所以：


$$
q(z_{t-1}\mid z_t,x)
\propto
q(z_t\mid z_{t-1})q(z_{t-1}\mid x).
$$


已知：


$$
q(z_t\mid z_{t-1})
=
N\left(
\sqrt{1-\beta_t}z_{t-1},
\beta_t I
\right).
$$


同时：


$$
q(z_{t-1}\mid x)
=
N\left(
\sqrt{\alpha_{t-1}}x,
(1-\alpha_{t-1})I
\right).
$$


为了简化，令：


$$
u=z_{t-1}.
$$


则第一个高斯项可以看成关于 $u$ 的函数：


$$
q(z_t\mid u)
\propto
\exp\left[
-
\frac{1}{2\beta_t}
\|z_t-\sqrt{1-\beta_t}u\|^2
\right].
$$


第二个高斯项为：


$$
q(u\mid x)
\propto
\exp\left[
-
\frac{1}{2(1-\alpha_{t-1})}
\|u-\sqrt{\alpha_{t-1}}x\|^2
\right].
$$


两者相乘，指数部分相加：


$$
\log q(u\mid z_t,x)
=
-
\frac{1}{2\beta_t}
\|z_t-\sqrt{1-\beta_t}u\|^2
-
\frac{1}{2(1-\alpha_{t-1})}
\|u-\sqrt{\alpha_{t-1}}x\|^2
+C.
$$


展开其中关于 $u$ 的二次项和一次项。

第一项：


$$
\|z_t-\sqrt{1-\beta_t}u\|^2
=
z_t^Tz_t
-2\sqrt{1-\beta_t}z_t^Tu
+(1-\beta_t)u^Tu.
$$


第二项：


$$
\|u-\sqrt{\alpha_{t-1}}x\|^2
=
u^Tu
-2\sqrt{\alpha_{t-1}}x^Tu
+\alpha_{t-1}x^Tx.
$$


只保留与 $u$ 有关的项：


$$
\log q(u\mid z_t,x)
=
-
\frac{1}{2}
\left[
\frac{1-\beta_t}{\beta_t}
+
\frac{1}{1-\alpha_{t-1}}
\right]u^Tu
+
\left[
\frac{\sqrt{1-\beta_t}}{\beta_t}z_t
+
\frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}x
\right]^T u
+C.
$$


这说明后验仍然是高斯分布：


$$
q(z_{t-1}\mid z_t,x)
=
N(\tilde{\mu}_t,\tilde{\beta}_t I).
$$


后验精度为：


$$
\tilde{\beta}_t^{-1}
=
\frac{1-\beta_t}{\beta_t}
+
\frac{1}{1-\alpha_{t-1}}.
$$


通分：


$$
\tilde{\beta}_t^{-1}
=
\frac{(1-\beta_t)(1-\alpha_{t-1})+\beta_t}
{\beta_t(1-\alpha_{t-1})}.
$$


注意：


$$
\alpha_t=\alpha_{t-1}(1-\beta_t).
$$


所以：


$$
(1-\beta_t)(1-\alpha_{t-1})+\beta_t
=1-\alpha_{t-1}+\alpha_{t-1}\beta_t+\beta_t
$$


更直接地整理为：


$$
(1-\beta_t)(1-\alpha_{t-1})+\beta_t
=1-\alpha_{t-1}(1-\beta_t)
=1-\alpha_t.
$$


因此：


$$
\tilde{\beta}_t^{-1}
=
\frac{1-\alpha_t}
{\beta_t(1-\alpha_{t-1})}.
$$


所以：


$$
\boxed{
\tilde{\beta}_t
=
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
}.
$$


后验均值为：


$$
\tilde{\mu}_t
=
\tilde{\beta}_t
\left[
\frac{\sqrt{1-\beta_t}}{\beta_t}z_t
+
\frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}x
\right].
$$


代入 $\tilde{\beta}_t$：


$$
\tilde{\mu}_t
=
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
\left[
\frac{\sqrt{1-\beta_t}}{\beta_t}z_t
+
\frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}x
\right].
$$


分别整理两项。

对于 $z_t$：


$$
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
\cdot
\frac{\sqrt{1-\beta_t}}{\beta_t}
=
\frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}.
$$


对于 $x$：


$$
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
\cdot
\frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}
=
\frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}.
$$


所以：


$$
\boxed{
\tilde{\mu}_t
=
\frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}x
+
\frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}z_t
}.
$$


最终：


$$
\boxed{
q(z_{t-1}\mid z_t,x)
=
N(\tilde{\mu}_t,\tilde{\beta}_t I)
}.
$$


其中：


$$
\boxed{
\tilde{\mu}_t
=
\frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}x
+
\frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}z_t
}.
$$



$$
\boxed{
\tilde{\beta}_t
=
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
}.
$$


---

## 图片题 2：线性高斯模型求 $p(z\mid x)$

### 题目

给定线性高斯模型：


$$
p(z)=N(0,I),
$$



$$
x=Wz+\mu+\epsilon,
$$


其中：


$$
\epsilon\sim N(0,\sigma^2I).
$$


求后验分布：


$$
p(z\mid x).
$$


### 解答

由模型：


$$
x=Wz+\mu+\epsilon,
\qquad
\epsilon\sim N(0,\sigma^2I),
$$


可得条件分布：


$$
p(x\mid z)
=
N(Wz+\mu,\sigma^2I).
$$


先验分布为：


$$
p(z)=N(0,I).
$$


根据贝叶斯公式：


$$
p(z\mid x)
\propto
p(x\mid z)p(z).
$$


写出与 $z$ 有关的指数部分：


$$
\log p(z\mid x)
=
-
\frac{1}{2\sigma^2}
\|x-\mu-Wz\|^2
-
\frac{1}{2}z^Tz
+C.
$$


令：


$$
r=x-\mu.
$$


则：


$$
\|x-\mu-Wz\|^2
=
\|r-Wz\|^2.
$$


展开：


$$
\|r-Wz\|^2
=
(r-Wz)^T(r-Wz).
$$



$$
=
r^Tr-2z^TW^Tr+z^TW^TWz.
$$


代回指数部分：


$$
\log p(z\mid x)
=
-
\frac{1}{2\sigma^2}
\left(
r^Tr-2z^TW^Tr+z^TW^TWz
\right)
-
\frac{1}{2}z^Tz
+C.
$$


忽略与 $z$ 无关的常数项：


$$
\log p(z\mid x)
=
-
\frac{1}{2\sigma^2}
z^TW^TWz
+
\frac{1}{\sigma^2}z^TW^Tr
-
\frac{1}{2}z^Tz
+C.
$$


把二次项合并：


$$
\log p(z\mid x)
=
-
\frac{1}{2}
z^T
\left(
I+\frac{1}{\sigma^2}W^TW
\right)
z
+
\frac{1}{\sigma^2}z^TW^T(x-\mu)
+C.
$$


这正是高斯分布的标准形式：


$$
\log p(z\mid x)
=
-
\frac{1}{2}
z^T\Sigma_{z\mid x}^{-1}z
+
z^T\Sigma_{z\mid x}^{-1}\mu_{z\mid x}
+C.
$$


因此后验精度矩阵为：


$$
\Sigma_{z\mid x}^{-1}
=
I+\frac{1}{\sigma^2}W^TW.
$$


所以后验协方差为：


$$
\boxed{
\Sigma_{z\mid x}
=
\left(
I+\frac{1}{\sigma^2}W^TW
\right)^{-1}
}.
$$


也可以写成：


$$
\boxed{
\Sigma_{z\mid x}
=
\sigma^2(W^TW+\sigma^2I)^{-1}
}.
$$


再比较一次项：


$$
\Sigma_{z\mid x}^{-1}\mu_{z\mid x}
=
\frac{1}{\sigma^2}W^T(x-\mu).
$$


所以：


$$
\mu_{z\mid x}
=
\Sigma_{z\mid x}
\frac{1}{\sigma^2}W^T(x-\mu).
$$


代入协方差表达式：


$$
\mu_{z\mid x}
=
\left(
I+\frac{1}{\sigma^2}W^TW
\right)^{-1}
\frac{1}{\sigma^2}W^T(x-\mu).
$$


也可以整理为：


$$
\boxed{
\mu_{z\mid x}
=
(W^TW+\sigma^2I)^{-1}W^T(x-\mu)
}.
$$


因此最终后验分布为：


$$
\boxed{
p(z\mid x)
=
N(\mu_{z\mid x},\Sigma_{z\mid x})
}.
$$


其中：


$$
\boxed{
\mu_{z\mid x}
=
(W^TW+\sigma^2I)^{-1}W^T(x-\mu)
}.
$$



$$
\boxed{
\Sigma_{z\mid x}
=
\sigma^2(W^TW+\sigma^2I)^{-1}
}.
$$



