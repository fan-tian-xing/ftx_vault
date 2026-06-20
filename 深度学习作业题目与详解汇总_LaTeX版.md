# 深度学习作业题目与详解汇总

---

# 第一次作业

## 一、香农熵

### 题目

在多类别分类问题中，某模型对 $K=4$ 个类别给出了以下概率预测分布：

$p=(0.5,0.25,0.125,0.125)^T$.

请计算该分布 $P$ 的香农熵 $H(P)$，使用 $\log_2$ 作为底，单位为比特。

### 解答

$H(P)=-\sum_i p_i\log_2 p_i$.

$H(P)=-(0.5\log_2 0.5+0.25\log_2 0.25+0.125\log_2 0.125+0.125\log_2 0.125)$.

$\log_2 0.5=-1$,

$\log_2 0.25=-2$,

$\log_2 0.125=-3$.

$0.5\log_2 0.5=0.5(-1)=-0.5$,

$0.25\log_2 0.25=0.25(-2)=-0.5$,

$0.125\log_2 0.125=0.125(-3)=-0.375$.

$H(P)=-[-0.5-0.5-0.375-0.375]$.

$H(P)=1.75$.

$\boxed{H(P)=1.75\text{ bits}}$.

---

## 二、多元高斯分布的 KL 散度

### 题目

设 $P$ 和 $Q$ 是 $n$ 维空间中的两个多元高斯分布：

$P\sim N(\mu_1,\Sigma_1),\qquad Q\sim N(\mu_2,\Sigma_2)$.

1. 写出 $D_{KL}(P\|Q)$ 的一般解析表达式。
2. 假设 $n=2$，$\mu_1=\mu_2=0$，$\Sigma_1=I$，$\Sigma_2=4I$，计算 $D_{KL}(P\|Q)$ 的值。

### 解答

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

$n=2,\qquad \mu_1=\mu_2=0,\qquad \Sigma_1=I,\qquad \Sigma_2=4I$.

$\Sigma_2^{-1}=(4I)^{-1}=\frac{1}{4}I$.

$\Sigma_2^{-1}\Sigma_1=\frac{1}{4}I\cdot I=\frac{1}{4}I$.

$$
\operatorname{tr}\left(\frac{1}{4}I\right)
=\frac{1}{4}+\frac{1}{4}
=\frac{1}{2}.
$$

$(\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)=0$.

$\det(\Sigma_1)=\det(I)=1$.

$\det(\Sigma_2)=\det(4I)$.

$\det(4I)=4^2=16$.

$$
\ln\frac{\det\Sigma_2}{\det\Sigma_1}
=\ln\frac{16}{1}
=\ln 16.
$$

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

$\frac{1}{2}\ln16=\ln4$，$D_{KL}(P\|Q)=\ln4-\frac{3}{4}$.

$\ln4\approx1.3863$，$D_{KL}(P\|Q)\approx1.3863-0.75=0.6363$.

$\boxed{D_{KL}(P\|Q)=\ln4-\frac{3}{4}\approx0.6363}$.

---

## 三、微分熵在变量变换下的关系

### 题目

设 $x$ 和 $y$ 是 $\mathbb{R}^n$ 上的连续随机向量，其概率密度函数分别为 $p_x(x)$ 和 $p_y(y)$。已知：

$y=f(x)$

是一个可微分且雅可比矩阵 $J_f(x)$ 非奇异的变换。

1. 证明随机变量 $y$ 的微分熵 $H(y)$ 与 $x$ 的微分熵 $H(x)$ 之间的关系满足：

$H(y)=H(x)-E_{p_y}[\ln|\det J_{f^{-1}}(y)|]$.

2. 假设变换是线性的：

$y=Wx$,

其中 $x$ 和 $y$ 均为 $n$ 维向量，$W$ 是一个 $n\times n$ 的可逆矩阵。矩阵 $W$ 需要满足什么条件，才能确保变换后的微分熵 $H(y)$ 小于原始熵 $H(x)$？

### 解答

$H(x)=-\int p_x(x)\ln p_x(x)\,dx$.

$p_y(y)=p_x(f^{-1}(y))\left|\det J_{f^{-1}}(y)\right|$.

$$
\ln p_y(y)
=\ln p_x(f^{-1}(y))
+\ln\left|\det J_{f^{-1}}(y)\right|.
$$

$H(y)=-E_{p_y}[\ln p_y(y)]$.

$$
H(y)
=-E_{p_y}
\left[
\ln p_x(f^{-1}(y))
+\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$

$$
H(y)
=-E_{p_y}[\ln p_x(f^{-1}(y))]
-E_{p_y}\left[
\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$

$$
-E_{p_y}[\ln p_x(f^{-1}(y))]
=-E_{p_x}[\ln p_x(x)]
=H(x).
$$

$$
H(y)=H(x)-E_{p_y}\left[
\ln\left|\det J_{f^{-1}}(y)\right|
\right].
$$

$$
\boxed{
H(y)=H(x)-E_{p_y}[\ln|\det J_{f^{-1}}(y)|]
}.
$$

$J_{f^{-1}}(y)=J_f(x)^{-1}$,

$\det J_{f^{-1}}(y)=\frac{1}{\det J_f(x)}$.

$$
\ln|\det J_{f^{-1}}(y)|
=-\ln|\det J_f(x)|.
$$

$H(y)=H(x)+E_{p_x}[\ln|\det J_f(x)|]$.

$J_f(x)=W$.

$H(y)=H(x)+\ln|\det W|$.

$H(y)<H(x)$,

$H(x)+\ln|\det W|<H(x)$.

$\ln|\det W|<0$.

$|\det W|<1$.

$\det W\neq0$.

$\boxed{0<|\det W|<1}$.

---

## 四、Softmax 与交叉熵求导

### 题目

在多分类问题中，Softmax 函数的预测概率为：

$a_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$.

损失函数使用交叉熵损失：

$L=-\sum_{k=1}^K y_k\ln a_k$,

其中 $y_k$ 是真实标签的 One-Hot 编码。请推导损失函数 $L$ 对 Softmax 输入 $z_i$ 的导数：

$\frac{\partial L}{\partial z_i}$.

### 解答

$a_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$.

先求 Softmax 的导数。对 $z_i$ 求导，有两种情况。

$$
\frac{\partial a_i}{\partial z_i}
=a_i(1-a_i).
$$

$$
\frac{\partial a_k}{\partial z_i}
=-a_ka_i.
$$

$$
\frac{\partial a_k}{\partial z_i}
=a_k(\delta_{ki}-a_i),
$$

$$
\delta_{ki}=
\begin{cases}
1, & k=i,\\
0, & k\neq i.
\end{cases}
$$

$L=-\sum_k y_k\ln a_k$.

$$
\frac{\partial L}{\partial a_k}
=-\frac{y_k}{a_k}.
$$

$$
\frac{\partial L}{\partial z_i}
=\sum_k
\frac{\partial L}{\partial a_k}
\frac{\partial a_k}{\partial z_i}.
$$

$$
\frac{\partial L}{\partial z_i}
=\sum_k
\left(-\frac{y_k}{a_k}\right)
a_k(\delta_{ki}-a_i).
$$

$$
\frac{\partial L}{\partial z_i}
=-\sum_k y_k(\delta_{ki}-a_i).
$$

$$
\frac{\partial L}{\partial z_i}
=-\sum_k y_k\delta_{ki}
+\sum_k y_ka_i.
$$

$\sum_k y_k\delta_{ki}=y_i$.

$$
\sum_k y_ka_i
=a_i\sum_k y_k.
$$

$\sum_k y_k=1$.

$\sum_k y_ka_i=a_i$.

$$
\frac{\partial L}{\partial z_i}
=-y_i+a_i.
$$

$$
\boxed{
\frac{\partial L}{\partial z_i}=a_i-y_i
}.
$$

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

$f(V;x)=Vx\in\mathbb{R}^C$,

其中：

$V=(v_1^T;\cdots;v_C^T)\in\mathbb{R}^{C\times d}$,

$v_i\in\mathbb{R}^d$ 是与第 $i$ 个输出相关的权重向量。

最小化均方误差损失函数：

$\ell_{MSE}(V)=\frac{1}{N}\sum_{n=1}^N \|Vx_n-Y_n\|_2^2$.

其中 $Y_n\in\mathbb{R}^C$ 是目标输出向量。计算损失函数关于 $V$ 的梯度，即：

$\frac{\partial \ell}{\partial v_i}$,

以及 Hessian 矩阵：

$\frac{\partial^2 \ell}{\partial v_i\partial v_j}$.

### 解答

$\hat{Y}_n=Vx_n$.

$\hat{Y}_{n,i}=v_i^T x_n$.

$$
\ell(V)
=\frac{1}{N}\sum_{n=1}^N
\sum_{i=1}^C
(v_i^T x_n-Y_{n,i})^2.
$$

$$
\frac{\partial \ell}{\partial v_i}
=\frac{1}{N}\sum_{n=1}^N
\frac{\partial}{\partial v_i}
(v_i^T x_n-Y_{n,i})^2.
$$

$e_{n,i}=v_i^T x_n-Y_{n,i}$.

$$
\frac{\partial}{\partial v_i}e_{n,i}^2
=2e_{n,i}\frac{\partial e_{n,i}}{\partial v_i}.
$$

$$
\frac{\partial e_{n,i}}{\partial v_i}
=\frac{\partial}{\partial v_i}(v_i^T x_n-Y_{n,i})
=x_n.
$$

$$
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n.
$$

$$
\boxed{
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n
}.
$$

$e_n=Vx_n-Y_n$.

$$
\boxed{
\nabla_V\ell
=\frac{2}{N}\sum_{n=1}^N
(Vx_n-Y_n)x_n^T
}.
$$

接下来计算 Hessian。

$$
\frac{\partial \ell}{\partial v_i}
=\frac{2}{N}\sum_{n=1}^N
(v_i^T x_n-Y_{n,i})x_n.
$$

$$
\frac{\partial^2 \ell}{\partial v_i^2}
=\frac{2}{N}\sum_{n=1}^N
x_nx_n^T.
$$

$$
\boxed{
\frac{\partial^2 \ell}{\partial v_i^2}
=\frac{2}{N}\sum_{n=1}^N x_nx_n^T
}.
$$

$v_i^Tx_n-Y_{n,i}$

$$
\boxed{
\frac{\partial^2 \ell}{\partial v_i\partial v_j}=0,\qquad i\neq j
}.
$$

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

$$
x=(x_1,x_2)^T
\quad\longmapsto\quad
\phi(x).
$$

$$
\phi(x_1,x_2)
=
(1,x_1,x_2,x_1^2,x_1x_2,x_2^2)^T.
$$

如果需要更复杂的非线性边界，也可以使用三阶或五阶多项式特征。

$$
\hat{y}_n
=\sigma(w^T\phi(x_n)+b),
$$

$\sigma(t)=\frac{1}{1+e^{-t}}$.

$z_n=w^T\phi(x_n)+b$.

$\hat{y}_n=\sigma(z_n)$.

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

$$
\frac{\partial J}{\partial z_n}
=\hat{y}_n-y_n.
$$

$z_n=w^T\phi(x_n)+b$,

$$
\frac{\partial z_n}{\partial w}=\phi(x_n),
\qquad
\frac{\partial z_n}{\partial b}=1.
$$

$$
\frac{\partial J}{\partial w}
=
\frac{1}{N}\sum_{n=1}^N
(\hat{y}_n-y_n)\phi(x_n)
+\lambda w.
$$

$$
\boxed{
\nabla_w J
=
\frac{1}{N}\Phi^T(\hat{y}-y)+\lambda w
}.
$$

$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac{1}{N}\sum_{n=1}^N(\hat{y}_n-y_n)
}.
$$

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

模型 A：无正则化或 $\lambda$ 极小，允许更大的权重，决策边界通常更弯曲，拟合能力更强，但更容易过拟合。

模型 B：$\lambda$ 较大，权重被明显惩罚，决策边界通常更平滑、更简单，泛化能力可能更好，但过强正则也可能欠拟合。

```text








```

---

## 三、两层隐藏层神经网络的反向传播

### 题目

使用 Python `sklearn.datasets.make_moons` 函数生成一个“双月亮”数据集。搭建含有 2 个隐藏层的神经网络，第一个隐藏层有 10 个神经元，第二个隐藏层有 20 个神经元，每层神经元使用 sigmoid 激活函数。损失函数使用二分类交叉熵：

$\ell(y,\hat{y})=-y\log(\hat{y})-(1-y)\log(1-\hat{y})$.

1. 利用反向传播算法，写出每层权重和偏置的梯度。
2. 当 $W^{(l)}$ 和 $b^{(l)}$ 都为 0 时，计算其对应的梯度值。

### 解答

$a^{(0)}=x$.

$z^{(1)}=W^{(1)}a^{(0)}+b^{(1)}$,

$a^{(1)}=\sigma(z^{(1)})$.

$z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}$,

$a^{(2)}=\sigma(z^{(2)})$.

$z^{(3)}=W^{(3)}a^{(2)}+b^{(3)}$,

$\hat{y}=a^{(3)}=\sigma(z^{(3)})$.

$$
\ell
=-y\ln\hat{y}-(1-y)\ln(1-\hat{y}).
$$

$$
\delta^{(3)}
=\frac{\partial \ell}{\partial z^{(3)}}
=\hat{y}-y.
$$

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

$$
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
\sigma'(z^{(2)}).
$$

$\sigma'(z)=\sigma(z)(1-\sigma(z))$,

$$
\sigma'(z^{(2)})
=a^{(2)}\odot(1-a^{(2)}).
$$

$$
\boxed{
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
a^{(2)}\odot(1-a^{(2)})
}.
$$

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

$$
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
\sigma'(z^{(1)}).
$$

$$
\boxed{
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
a^{(1)}\odot(1-a^{(1)})
}.
$$

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

$$
\frac{\partial L}{\partial W^{(l)}}
=
\frac{1}{N}\sum_{n=1}^N
\delta_n^{(l)}(a_n^{(l-1)})^T.
$$

下面讨论所有 $W^{(l)}$ 和 $b^{(l)}$ 都为 0 的情况。

$z^{(1)}=0$,

$a^{(1)}=\sigma(0)=0.5$.

这里 $a^{(1)}$ 是 10 维向量，所有分量都为 $0.5$。

$z^{(2)}=0,\qquad a^{(2)}=\sigma(0)=0.5$.

这里 $a^{(2)}$ 是 20 维向量，所有分量都为 $0.5$。

$z^{(3)}=0$,

$\hat{y}=a^{(3)}=\sigma(0)=0.5$.

$\delta^{(3)}=\hat{y}-y=0.5-y$.

$$
\boxed{
\frac{\partial \ell}{\partial b^{(3)}}=0.5-y
}.
$$

$$
\frac{\partial \ell}{\partial W^{(3)}}
=\delta^{(3)}(a^{(2)})^T.
$$

$$
\boxed{
\frac{\partial \ell}{\partial W^{(3)}}
=(0.5-y)(0.5,\ldots,0.5)
}.
$$

$$
\delta^{(2)}
=
\left((W^{(3)})^T\delta^{(3)}\right)
\odot
a^{(2)}\odot(1-a^{(2)}).
$$

$W^{(3)}=0$,

$(W^{(3)})^T\delta^{(3)}=0$.

$\boxed{\delta^{(2)}=0}$.

$$
\boxed{
\frac{\partial \ell}{\partial W^{(2)}}=0
},
\qquad
\boxed{
\frac{\partial \ell}{\partial b^{(2)}}=0
}.
$$

$$
\delta^{(1)}
=
\left((W^{(2)})^T\delta^{(2)}\right)
\odot
a^{(1)}\odot(1-a^{(1)}).
$$

$\delta^{(2)}=0$,

$\boxed{\delta^{(1)}=0}$.

$$
\boxed{
\frac{\partial \ell}{\partial W^{(1)}}=0
},
\qquad
\boxed{
\frac{\partial \ell}{\partial b^{(1)}}=0
}.
$$

当所有权重和偏置都初始化为 0 时，只有输出层的梯度可能非零，隐藏层梯度为零。这说明全零初始化会导致隐藏层无法有效学习，因此实际训练神经网络时通常不能把所有权重都初始化为 0。

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

$(3-2+1)\times(3-2+1)=2\times2$.

$$
K=
\begin{pmatrix}
1 & 1\\
-1 & 1
\end{pmatrix}.
$$

$$
\begin{pmatrix}
2 & 1\\
4 & 3
\end{pmatrix}.
$$

$$
2\cdot1+1\cdot1+4\cdot(-1)+3\cdot1
=2+1-4+3=2.
$$

$$
\begin{pmatrix}
1 & 1\\
3 & 5
\end{pmatrix}.
$$

$$
1\cdot1+1\cdot1+3\cdot(-1)+5\cdot1
=1+1-3+5=4.
$$

$$
\begin{pmatrix}
4 & 3\\
7 & 6
\end{pmatrix}.
$$

$$
4\cdot1+3\cdot1+7\cdot(-1)+6\cdot1
=4+3-7+6=6.
$$

$$
\begin{pmatrix}
3 & 5\\
6 & 0
\end{pmatrix}.
$$

$$
3\cdot1+5\cdot1+6\cdot(-1)+0\cdot1
=3+5-6+0=2.
$$

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

$W_1+2P$.

卷积核宽度为 $F$，因此第一个卷积窗口占据 $F$ 个位置。若步长为 $S$，窗口每次移动 $S$ 个单位。

$W_1+2P-F$.

$\left\lfloor \frac{W_1-F+2P}{S}\right\rfloor+1$

个位置。

$$
\boxed{
W_2=
\left\lfloor
\frac{W_1-F+2P}{S}
\right\rfloor+1
}.
$$

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

$C(T_{a,b}\circ x)=T_{a,b}\circ C(x)$,

$C(x)=x*w$,

$(T_{a,b}\circ x)_{i,j}=x_{i-a,j-b}$.

### 解答

要证明两个矩阵相等，只需要证明它们在任意位置 $(i,j)$ 的元素相等。

$\left(C(T_{a,b}\circ x)\right)_{i,j}$.

$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
\sum_{u,v}
(T_{a,b}\circ x)_{i-u,j-v}w_{u,v}.
$$

$$
(T_{a,b}\circ x)_{i-u,j-v}
=x_{i-u-a,j-v-b}.
$$

$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$

$(T_{a,b}\circ C(x))_{i,j}$.

$$
(T_{a,b}\circ C(x))_{i,j}
=C(x)_{i-a,j-b}.
$$

$$
C(x)_{i-a,j-b}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$

$$
(T_{a,b}\circ C(x))_{i,j}
=
\sum_{u,v}
x_{i-a-u,j-b-v}w_{u,v}.
$$

$$
\left(C(T_{a,b}\circ x)\right)_{i,j}
=
(T_{a,b}\circ C(x))_{i,j}.
$$

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

$H(x)$.

$F(x)=H(x)-x$.

$H(x)=F(x)+x$.

$$
\boxed{
y=F(x)+x
}.
$$

$y\approx x$.

这比让若干非线性层直接学习恒等映射更容易。

1. 缓解深层网络退化问题。
2. 梯度可以通过 shortcut 更容易向前层传播。
3. 使非常深的网络更容易优化。
4. 让网络学习残差变化，而不是完整映射，降低优化难度。

$\mu_B=\frac{1}{m}\sum_{i=1}^m x_i$,

$\sigma_B^2=\frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2$.

$\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$.

$y_i=\gamma \hat{x}_i+\beta$.

1. 稳定每层输入分布，使训练更平稳。
2. 缓解梯度消失或梯度爆炸。
3. 加快网络收敛速度。
4. 使残差分支 $F(x)$ 的数值尺度更加稳定，方便与 shortcut 分支 $x$ 相加。
5. 对深层网络尤其重要，因为网络越深，激活分布和梯度尺度越容易变得不稳定。

ResNet 通常将卷积层、BN 层和激活函数组合使用，使残差分支更容易被优化。

---

## 二、循环神经网络反向传播

### 题目

考虑有两个输入 $X_1$ 和 $X_2$，一个输出 $Y$，两个隐藏状态 $h_1,h_2$ 的循环神经网络：

$h_1=\tanh(Wh_0+UX_1+b)$,

$h_2=\tanh(Wh_1+UX_2+b)$,

$Y=Vh_2+c$.

假设所有参数都是一维的，损失函数为：

$L=\frac{1}{2}(Y-Z)^2$,

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

$s_1=Wh_0+UX_1+b$,

$h_1=\tanh(s_1)$.

$s_2=Wh_1+UX_2+b$,

$h_2=\tanh(s_2)$.

$Y=Vh_2+c$.

$L=\frac{1}{2}(Y-Z)^2$.

$e=Y-Z$.

$\frac{\partial L}{\partial Y}=Y-Z=e$.

先求输出层参数 $V,c$。

$Y=Vh_2+c$,

$$
\frac{\partial Y}{\partial V}=h_2,
\qquad
\frac{\partial Y}{\partial c}=1.
$$

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

$$
\frac{\partial L}{\partial h_2}
=
\frac{\partial L}{\partial Y}
\frac{\partial Y}{\partial h_2}
=eV.
$$

$h_2=\tanh(s_2)$,

$\frac{d}{ds}\tanh(s)=1-\tanh^2(s)$,

$$
\frac{\partial h_2}{\partial s_2}
=1-h_2^2.
$$

$g_2=\frac{\partial L}{\partial s_2}$.

$$
g_2=
\frac{\partial L}{\partial h_2}
\frac{\partial h_2}{\partial s_2}
=eV(1-h_2^2).
$$

$$
\boxed{
g_2=eV(1-h_2^2)
}.
$$

再把误差传到第一个时间步。

$s_2=Wh_1+UX_2+b$,

$\frac{\partial s_2}{\partial h_1}=W$.

$$
\frac{\partial L}{\partial h_1}
=
\frac{\partial L}{\partial s_2}
\frac{\partial s_2}{\partial h_1}
=g_2W.
$$

$h_1=\tanh(s_1)$,

$$
\frac{\partial h_1}{\partial s_1}
=1-h_1^2.
$$

$g_1=\frac{\partial L}{\partial s_1}$.

$$
g_1
=
\frac{\partial L}{\partial h_1}
\frac{\partial h_1}{\partial s_1}
=g_2W(1-h_1^2).
$$

$$
\boxed{
g_1=g_2W(1-h_1^2)
}.
$$

现在计算共享参数 $W,U,b$ 的梯度。由于 $W,U,b$ 在两个时间步中共享，所以总梯度等于两个时间步贡献之和。

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

$$
\boxed{
\frac{\partial L}{\partial W}
=g_1h_0+g_2h_1
}.
$$

$$
\frac{\partial s_1}{\partial U}=X_1,
\qquad
\frac{\partial s_2}{\partial U}=X_2.
$$

$$
\boxed{
\frac{\partial L}{\partial U}
=g_1X_1+g_2X_2
}.
$$

$$
\frac{\partial s_1}{\partial b}=1,
\qquad
\frac{\partial s_2}{\partial b}=1.
$$

$$
\boxed{
\frac{\partial L}{\partial b}
=g_1+g_2
}.
$$

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

$B=PAP^T$

$B_{ij}=A_{\pi(i),\pi(j)}$.

$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{B_{ij}}}{\sum_{k=1}^N e^{B_{ik}}}.
$$

$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{A_{\pi(i),\pi(j)}}}
{\sum_{k=1}^N e^{A_{\pi(i),\pi(k)}}}.
$$

$$
\sum_{k=1}^N e^{A_{\pi(i),\pi(k)}}
=
\sum_{m=1}^N e^{A_{\pi(i),m}}.
$$

$$
\operatorname{Softmax}(B)_{ij}
=
\frac{e^{A_{\pi(i),\pi(j)}}}
{\sum_{m=1}^N e^{A_{\pi(i),m}}}.
$$

$\operatorname{Softmax}(A)_{\pi(i),\pi(j)}$.

$P\operatorname{Softmax}(A)P^T$

$\operatorname{Softmax}(A)_{\pi(i),\pi(j)}$.

$$
\operatorname{Softmax}(PAP^T)_{ij}
=
\left(P\operatorname{Softmax}(A)P^T\right)_{ij}.
$$

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

$\operatorname{Att}(PX)=P\operatorname{Att}(X)$.

自注意力模块对输入 token 的置换是等变的。

2. 分析自注意力模型和卷积神经网络之间的关系。

### 解答

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$

$$
\operatorname{Att}(X)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
$$

$Q'=(PX)W_Q=P(XW_Q)=PQ$.

$K'=(PX)W_K=P(XW_K)=PK$.

$V'=(PX)W_V=P(XW_V)=PV$.

$$
Q'(K')^T
=(PQ)(PK)^T.
$$

$(PK)^T=K^TP^T$,

$$
Q'(K')^T
=P QK^T P^T.
$$

$$
\frac{Q'(K')^T}{\sqrt{d_k}}
=
P
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
P^T.
$$

$$
\operatorname{Softmax}(PAP^T)
=P\operatorname{Softmax}(A)P^T.
$$

$A=\frac{QK^T}{\sqrt{d_k}}$,

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

$$
\operatorname{Att}(PX)
=
\operatorname{Softmax}
\left(
\frac{Q'(K')^T}{\sqrt{d_k}}
\right)
V'.
$$

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

$P^TP=I$,

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

$$
\operatorname{Att}(X)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V.
$$

$$
\boxed{
\operatorname{Att}(PX)=P\operatorname{Att}(X)
}.
$$

这说明：如果输入 token 的顺序被同一个置换矩阵 $P$ 重排，那么输出 token 也会以相同方式重排。因此自注意力本身具有置换等变性。

下面分析自注意力与 CNN 的关系。

$y_i=\sum_{j\in\mathcal{N}(i)} w_{i-j}x_j$.

$\mathcal{N}(i)$ 是位置 $i$ 附近的局部邻域，$w_{i-j}$ 是卷积核权重。

1. 权重是固定的，训练完成后与具体输入内容无关。
2. 聚合范围通常是局部邻域。
3. 卷积核在不同空间位置共享。
4. 具有平移等变性。

$y_i=\sum_{j=1}^N \alpha_{ij}v_j$.

$$
\alpha_{ij}
=
\frac{\exp(q_i^Tk_j/\sqrt{d_k})}
{\sum_m \exp(q_i^Tk_m/\sqrt{d_k})}.
$$

1. 权重 $\alpha_{ij}$ 由输入内容动态计算。
2. 每个 token 可以和所有 token 建立联系，因此天然具有全局建模能力。
3. 权重不是固定卷积核，而是随样本变化。
4. 如果不加入位置编码，自注意力对 token 顺序没有内置位置感。

二者的共同点是：输出本质上都是对输入特征的加权求和。

区别是：CNN 使用固定的、局部的、位置共享的权重；自注意力使用动态的、内容相关的、通常全局的权重。

自注意力可以看作一种更灵活的动态卷积。它能够在一定条件下模拟卷积操作，但表达能力更依赖输入内容和注意力权重。

---

## 五、图拉普拉斯公式证明

### 题目

设：

$x=(x_1,x_2,\ldots,x_N)^T\in\mathbb{R}^N$

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

$L=D-A$

表示图拉普拉斯矩阵，$D$ 表示度矩阵，$A$ 表示连接矩阵。

### 解答

$L=D-A$.

$x^TLx=x^T(D-A)x$.

$x^TLx=x^TDx-x^TAx$.

$D_{ii}=d_i$.

$x^TDx=\sum_{i=1}^N d_ix_i^2$.

$d_i=\sum_{j=1}^N A_{ij}$.

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

$$
x^TAx
=
\sum_{i=1}^N
\sum_{j=1}^N
A_{ij}x_ix_j.
$$

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

$$
x^TLx
=
\sum_{i,j}A_{ij}x_i^2
-
\sum_{i,j}A_{ij}x_ix_j.
$$

$$
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i-x_j)^2.
$$

$(x_i-x_j)^2=x_i^2-2x_ix_j+x_j^2$,

$$
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i-x_j)^2
=
\frac{1}{2}
\sum_{i,j}
A_{ij}(x_i^2-2x_ix_j+x_j^2).
$$

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

$A_{ij}=A_{ji}$.

$$
\sum_{i,j}A_{ij}x_j^2
=
\sum_{i,j}A_{ji}x_i^2
=
\sum_{i,j}A_{ij}x_i^2.
$$

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

$x^TLx$.

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

$\alpha_t=\prod_{s=1}^t(1-\beta_s)$.

求：

$q(z_{t-1}\mid z_t,x)$.

### 解答

$q(z_{t-1}\mid z_t,x)$.

$$
q(z_{t-1}\mid z_t,x)
\propto
q(z_t\mid z_{t-1},x)q(z_{t-1}\mid x).
$$

$q(z_t\mid z_{t-1},x)=q(z_t\mid z_{t-1})$.

$$
q(z_{t-1}\mid z_t,x)
\propto
q(z_t\mid z_{t-1})q(z_{t-1}\mid x).
$$

$$
q(z_t\mid z_{t-1})
=
N\left(
\sqrt{1-\beta_t}z_{t-1},
\beta_t I
\right).
$$

$$
q(z_{t-1}\mid x)
=
N\left(
\sqrt{\alpha_{t-1}}x,
(1-\alpha_{t-1})I
\right).
$$

$u=z_{t-1}$.

$$
q(z_t\mid u)
\propto
\exp\left[
-
\frac{1}{2\beta_t}
\|z_t-\sqrt{1-\beta_t}u\|^2
\right].
$$

$$
q(u\mid x)
\propto
\exp\left[
-
\frac{1}{2(1-\alpha_{t-1})}
\|u-\sqrt{\alpha_{t-1}}x\|^2
\right].
$$

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

$$
\|z_t-\sqrt{1-\beta_t}u\|^2
=
z_t^Tz_t
-2\sqrt{1-\beta_t}z_t^Tu
+(1-\beta_t)u^Tu.
$$

$$
\|u-\sqrt{\alpha_{t-1}}x\|^2
=
u^Tu
-2\sqrt{\alpha_{t-1}}x^Tu
+\alpha_{t-1}x^Tx.
$$

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

$$
q(z_{t-1}\mid z_t,x)
=
N(\tilde{\mu}_t,\tilde{\beta}_t I).
$$

$$
\tilde{\beta}_t^{-1}
=
\frac{1-\beta_t}{\beta_t}
+
\frac{1}{1-\alpha_{t-1}}.
$$

$$
\tilde{\beta}_t^{-1}
=
\frac{(1-\beta_t)(1-\alpha_{t-1})+\beta_t}
{\beta_t(1-\alpha_{t-1})}.
$$

$\alpha_t=\alpha_{t-1}(1-\beta_t)$.

$$
(1-\beta_t)(1-\alpha_{t-1})+\beta_t
=1-\alpha_{t-1}+\alpha_{t-1}\beta_t+\beta_t
$$

$$
(1-\beta_t)(1-\alpha_{t-1})+\beta_t
=1-\alpha_{t-1}(1-\beta_t)
=1-\alpha_t.
$$

$$
\tilde{\beta}_t^{-1}
=
\frac{1-\alpha_t}
{\beta_t(1-\alpha_{t-1})}.
$$

$$
\boxed{
\tilde{\beta}_t
=
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
}.
$$

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

$$
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
\cdot
\frac{\sqrt{1-\beta_t}}{\beta_t}
=
\frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}.
$$

$$
\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t
\cdot
\frac{\sqrt{\alpha_{t-1}}}{1-\alpha_{t-1}}
=
\frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}.
$$

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
q(z_{t-1}\mid z_t,x)
=
N(\tilde{\mu}_t,\tilde{\beta}_t I)
}.
$$

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

$p(z)=N(0,I)$,

$x=Wz+\mu+\epsilon$,

其中：

$\epsilon\sim N(0,\sigma^2I)$.

求后验分布：

$p(z\mid x)$.

### 解答

$$
x=Wz+\mu+\epsilon,
\qquad
\epsilon\sim N(0,\sigma^2I),
$$

$$
p(x\mid z)
=
N(Wz+\mu,\sigma^2I).
$$

$p(z)=N(0,I)$.

$$
p(z\mid x)
\propto
p(x\mid z)p(z).
$$

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

$r=x-\mu$.

$$
\|x-\mu-Wz\|^2
=
\|r-Wz\|^2.
$$

$$
\|r-Wz\|^2
=
(r-Wz)^T(r-Wz).
$$

$$
=
r^Tr-2z^TW^Tr+z^TW^TWz.
$$

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

$$
\Sigma_{z\mid x}^{-1}
=
I+\frac{1}{\sigma^2}W^TW.
$$

$$
\boxed{
\Sigma_{z\mid x}
=
\left(
I+\frac{1}{\sigma^2}W^TW
\right)^{-1}
}.
$$

$$
\boxed{
\Sigma_{z\mid x}
=
\sigma^2(W^TW+\sigma^2I)^{-1}
}.
$$

$$
\Sigma_{z\mid x}^{-1}\mu_{z\mid x}
=
\frac{1}{\sigma^2}W^T(x-\mu).
$$

$$
\mu_{z\mid x}
=
\Sigma_{z\mid x}
\frac{1}{\sigma^2}W^T(x-\mu).
$$

$$
\mu_{z\mid x}
=
\left(
I+\frac{1}{\sigma^2}W^TW
\right)^{-1}
\frac{1}{\sigma^2}W^T(x-\mu).
$$

$$
\boxed{
\mu_{z\mid x}
=
(W^TW+\sigma^2I)^{-1}W^T(x-\mu)
}.
$$

$$
\boxed{
p(z\mid x)
=
N(\mu_{z\mid x},\Sigma_{z\mid x})
}.
$$

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

---

# 第12章第一题

### 题目

如果 $\operatorname{Var}(x_{t-1})=I$，并且

$$
x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t,
$$

其中，随机变量 $x_{t-1}$ 与 $\epsilon_t$ 独立。证明

$$
\operatorname{Var}(x_t)=I.
$$

这说明每个时刻 $x_t$ 方差保持相等。

### 解答

由题意，

$$
x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t.
$$

要计算 $x_t$ 的方差，使用方差的线性变换公式。若随机变量 $X$ 与 $Y$ 独立，则

$$
\operatorname{Var}(aX+bY)
=
a^2\operatorname{Var}(X)
+
b^2\operatorname{Var}(Y).
$$

在本题中，

$$
a=\sqrt{1-\beta_t},
\qquad
b=\sqrt{\beta_t}.
$$

因此，

$$
\operatorname{Var}(x_t)
=
\operatorname{Var}
\left(
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t
\right).
$$

由于 $x_{t-1}$ 与 $\epsilon_t$ 独立，交叉协方差项为 $0$，所以

$$
\operatorname{Var}(x_t)
=
(1-\beta_t)\operatorname{Var}(x_{t-1})
+
\beta_t\operatorname{Var}(\epsilon_t).
$$

扩散模型中的噪声 $\epsilon_t$ 通常为标准高斯噪声，即

$$
\epsilon_t\sim N(0,I),
\qquad
\operatorname{Var}(\epsilon_t)=I.
$$

又因为题目给出

$$
\operatorname{Var}(x_{t-1})=I,
$$

代入可得

$$
\operatorname{Var}(x_t)
=
(1-\beta_t)I+\beta_t I.
$$

合并同类项：

$$
\operatorname{Var}(x_t)
=
\left(1-\beta_t+\beta_t\right)I.
$$

因此

$$
\boxed{
\operatorname{Var}(x_t)=I
}.
$$

这说明如果上一时刻 $x_{t-1}$ 的方差为单位矩阵，并且每一步加入的噪声也是单位方差噪声，那么经过

$$
x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t
$$

这样的更新之后，$x_t$ 的方差仍然保持为 $I$。也就是说，每个时刻的 $x_t$ 都保持相同的方差。
