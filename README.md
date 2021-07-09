# **高等代数 & 计算机图形学**

***
# 网络资料
***
[闫令琪CG公开课首页](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
[闫令琪CG公开课作业](http://games-cn.org/forums/topic/allhw/)
[CSDN大佬整理的闫令琪CG公开课笔记](https://blog.csdn.net/qq_38065509/article/details/105156501)
[大佬整理的《Polygon Mesh Processing》阅读笔记](https://www.jianshu.com/p/47c350770ffe)




***
***
***
# 暂时不知道如何分类
***
## 一些向量代数公式
*	混合积
	*	表示以$\overrightarrow{a},\overrightarrow{b},\overrightarrow{c}$为邻边的六面体的体积。
$$
(\overrightarrow{a},\overrightarrow{b},\overrightarrow{c})=(\overrightarrow{a}\times \overrightarrow{b})\cdot \overrightarrow{c}
$$
*	双重外积公式
$$
\overrightarrow{a}\times \overrightarrow{b}\times \overrightarrow{c} = (\overrightarrow{a}\cdot\overrightarrow{c})\overrightarrow{b}-(\overrightarrow{b}\cdot\overrightarrow{c})\overrightarrow{a}
$$
*	拉格朗日恒等式
$$
(\overrightarrow{a}\times \overrightarrow{b})\cdot (\overrightarrow{c}\times \overrightarrow{d}) = (\overrightarrow{a}\cdot\overrightarrow{c})(\overrightarrow{b}\cdot\overrightarrow{d})-(\overrightarrow{a}\cdot\overrightarrow{d})(\overrightarrow{b}\cdot\overrightarrow{c})
$$

*	向量$\overrightarrow{b}$在向量$\overrightarrow{a}$方向上的投影：
$$
\overrightarrow{b_{proj}} = (\overrightarrow{b}\cdot \hat{a})\hat{a}
$$
*	向量$\overrightarrow{b}$在以向量$\overrightarrow{a}$为法线的平面上的投影：
$$
\overrightarrow{b_{proj}} = \overrightarrow{b}-(\overrightarrow{b}\cdot \hat{a})\hat{a}
$$



## 流形网格(manifold mesh)
*	满足以下条件的都是流形网格，不满足的都是非流形网格：
	*	一条边为一个或两个三角片共享。
	*	一个顶点的一环领域三角片构成一个开放或闭合的扇面。	

![](.\images\manifold.png)
![](.\images\non-manifolds.png)



## 环

## 域

## 群

## 欧拉角(Euler angles)

* 用来确定定点转动刚体位置的3个一组独立角参量。
* 共有三个欧拉角——章动角θ、旋进角（即进动角）ψ、自转角φ。
* 欧拉角是最常见的用三个数值来表示旋转的办法，因为比较直观容易想象。
	* 用先后三次当前围绕坐标系轴的旋转（三个自由度为一的旋转）来表示一个三维的自由旋转。



## 四元数(Quaternion) 

* 欧拉角和四元数都可以描述旋转，但是欧拉角有万向节死锁问题，所以四元数用得更普遍一些。
* 由实数加上三个虚数组成：
$$
q = w+xi+yj+zk = s+\overrightarrow{v}\\
$$
*	$ i, j, k $是虚数单位，它们有如下关系： 
$$
i^2 = j^2 = k^2 = ijk = -1\\
i = jk = -kj \\
j = ki = -ik \\
k = ij = -jk \\
$$
*	四元数的运算法则：
	*	乘积		$ q_1q_2 = (s_1s_2-\overrightarrow{v_1}\overrightarrow{v_2}+s_1\overrightarrow{v_2}+s_2\overrightarrow{v_1}+\overrightarrow{v_1}\times\overrightarrow{v_2}) $
	*	共轭		$ q^* =  s-\overrightarrow{v} $
	*	逆		$ q^{-1} = \frac{q^*}{|q|^2}$
	*	绝对值		$ |q| = \sqrt{qq^*} = \sqrt{w^2+x^2+y^2+z^2} $
*	对于$ i, j, k $本身的几何意义可以理解为一种旋转
	* i旋转代表X轴与Y轴相交平面中X轴正向向Y轴正向的旋转。
	
	* j旋转代表Z轴与X轴相交平面中Z轴正向向X轴正向的旋转。
	
	* k旋转代表Y轴与Z轴相交平面中Y轴正向向Z轴正向的旋转。
	
	* $ -i, -j, -k $分别代表$ i, j, k $旋转的反向旋转。
*	四元数$q$和旋转轴（单位向量为$\overrightarrow{u}$）、旋转角（$\theta$）的关系：
$$
q = (s, \overrightarrow{v})\\
	s = cos{\frac{\theta}{2}}\\
	\overrightarrow{v} = \overrightarrow{u} sin{\frac{\theta}{2}}
$$
*	四元数描述的三维旋转——$P = (0,\overrightarrow{p})$是一个四元数表示的三维空间向量，被四元数q施加旋转操作后结果为：
$$
P'=qPq^{-1}\\
\begin{aligned}
\overrightarrow{p'} &= s^2\overrightarrow{p}+\overrightarrow{v}(\overrightarrow{p}\cdot\overrightarrow{v})+2s(\overrightarrow{v}\times \overrightarrow{p})+ \overrightarrow{v}\times (\overrightarrow{v}\times \overrightarrow{p})\\
&= \overrightarrow{p}+2s(\overrightarrow{v}\times \overrightarrow{p})+ 2\overrightarrow{v}\times (\overrightarrow{v}\times \overrightarrow{p})
\end{aligned}
$$




## 旋转
* 一个旋转序列等价于一个单个旋转。
*	两类旋转问题：
	*	坐标系固定，代表物体的某点（或向量）以某向量为轴旋转某角度，求旋转后的点坐标。
	*	物体不变，坐标系以某向量为轴旋转某角度，求旋转后点在新坐标系下的坐标。
*	常见应用场景。
	*	两套坐标系：全局坐标、局部坐标。
	*	局部坐标以物体自身中心为坐标原点。
	*	联系全局、局部坐标系的两个量
		*	向量center
		*	四元数q
		*	局部坐标系中的某点经过q旋转，然后沿着center平移之后，到达的位置的坐标，等于该点在全局坐标系中的坐标。



### 坐标系
**左手、右手坐标系**
![](.\images\coordinate-system.jpg)

**旋转正方向**

*	对于左手坐标系而言，确定一个旋转轴后，左手握住拳头，拇指指向旋转轴的正方向，四指弯曲的方向为旋转的正方向。


### 变换矩阵
*	旋转变换矩阵是正交阵，即矩阵的逆等于矩阵的转置。

**二维旋转**

*	二维旋转是绕一点顺时针或逆时针旋转，逆时针为正
*	绕原点逆时针旋转$\theta$角的变换矩阵：
$$
R(\theta) = \begin{bmatrix}cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}\\
$$

**三维旋转**

*	三维旋转以某一条直线为轴旋转。
*	绕x, y, z轴旋转$\theta$角的变换矩阵：
$$
R_x(\theta) = \begin{bmatrix}1 & 0 & 0 \\ 0 & cos\theta & -sin\theta \\ 0 & sin\theta & cos\theta \end{bmatrix} \\ \\
R_y(\theta) = \begin{bmatrix}cos\theta & 0 & sin\theta \\ 0 & 1 & 0 \\ -sin\theta & 0 & cos\theta \end{bmatrix} \\ \\
R_z(\theta) =\begin{bmatrix}cos\theta & -sin\theta & 0 \\ sin\theta & cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \\
$$















***
***
***
# 线性方程组
$$
A_{m\times n}\overrightarrow{x}_{n\times 1} = \overrightarrow{\beta}_{m\times 1}
$$
*	分类
	*	按照$\overrightarrow{\beta}$向量是否为向量：
		1. 齐次：$\overrightarrow{\beta}$为零向量
		2. 非齐次：$\overrightarrow{\beta}$不为零向量
	*	按照方程个数$m$和未知数个数$n$的比较：
		1. 欠定：$m<n$
		2. 恰定：$m=n$
		3. 超定：$m>n$



## 齐次线性方程组(HLE: homogeneous linear equations)
$$
A_{m\times n}\overrightarrow{x}_{n\times 1} = 0
$$
*	常数项全为0的线性方程组，$m$为方程数，$n$为元数。
*	解的情况：
	*	$m<n$：有无穷个非零解和全零解。
	*	$m≥n$：只有全零解。
*	齐次线性方程组的非零解：
	*	无穷个非零解构成一个$n-r$维的向量空间。
	*	就是矩阵$A$对应的线性变换的零空间。



## 非齐次线性方程组(NLE: non-homogeneous linear equations)
$$
A_{m\times n}\overrightarrow{x}_{n\times 1} = \overrightarrow{\beta}_{m\times 1}\\
B = (A, \overrightarrow{\beta})
$$
* $A$为线性方程组的系数矩阵，$B$为增广矩阵。

*	解的情况：
	*	无解 ⇔ $R(A)<R(B)$
	*	有唯一解 ⇔ $R(A)=R(B)=n$
	*	有无穷多解 ⇔ $R(A)=R(B)<n$
	
* NLE的通解 = HLE的通解 + NLE的一个特解

* 从几何的角度看

  * HLE的解集经特解向量$\overrightarrow{p}$平移得到NLE的解集。

*	从线性映射的角度看
	*	$A$表示从$V^m$映射到$V^n$空间的一个线性映射。
	*	解这样一个线性方程组本质上是求在$A$代表的线性映射下，向量空间$V^m$中什么样的向量$\overrightarrow{x}$可以被映射成$V^n$空间中的$\overrightarrow{\beta}$

*	特解$\overrightarrow{p}$为$rref(B)$的最后一列。$rref(B)$为B经过Gauss-Jordan消元法得到的简化行阶梯型矩阵。














***
***
***
# 矩阵性质、运算、关系
## 转置
$$
(AB)^T = B^TA^T
$$

## 共轭转置(Hermite共轭)
*	记为$A^*$或$A^+$
*	转置，对元素取复共轭$A^* = \bar{A}^T$



## 矩阵的秩(rank)
$$
\begin{aligned}
&方阵A满秩\\
\Leftrightarrow &A, A^T, AA^T都可逆\\
\Leftrightarrow &det(A)\ne 0\\
\Leftrightarrow &A非奇异\\
\Leftrightarrow &齐次线性方程组A\overrightarrow{x}=0只有零解。
&\end{aligned}
$$



![](.\images\relations_of_matrices.png)
## 矩阵相抵/等价(equivalent)




## 矩阵相似(similar)
*	$n$级矩阵$A$和$B$相似 $\Leftrightarrow A = P^{-1}BP $ ，其中$P$为$n$级可逆矩阵



## 可对角化矩阵(diagonalizable)
*	矩阵$P$经过矩阵$A$作用对角化为矩阵$Q$，$A$的所有列向量本质上是$P$的一个基，代表了$P$的特征空间，$P$代表的线性映射对这组基向量的变换只是伸缩变换，没有旋转。
$$
\begin{aligned}\\
&\qquad \quad A^{-1}PA=Q\\
其中： &\qquad \quad A = \{\overrightarrow{\alpha ^T_1}, \overrightarrow{\alpha ^T_2}, ..., \overrightarrow{\alpha ^T_n}\}\\
&\qquad \quad Q = diag\{\lambda_1, \lambda_2,..., \lambda_n\}\\
&\Rightarrow \qquad	PA=AQ\\
&\Rightarrow	\qquad	P\begin{bmatrix} \overrightarrow{\alpha ^T_1} & \overrightarrow{\alpha ^T_2} & ...& \overrightarrow{\alpha ^T_n} \end{bmatrix}  =  \begin{bmatrix}\overrightarrow{\alpha ^T_1} & \overrightarrow{\alpha ^T_2} & ...& \overrightarrow{\alpha ^T_n} \end{bmatrix}  \begin{bmatrix} \lambda_1 & 0 & 0 & ... \\ 0 & \lambda_2 & 0 & ...\\ ... & ... & ... & ...\\ ... & ... & ... & \lambda_n \end{bmatrix} \\
&\Rightarrow	\qquad	P\begin{bmatrix} \overrightarrow{\alpha ^T_1} & \overrightarrow{\alpha ^T_2} & ...& \overrightarrow{\alpha ^T_n} \end{bmatrix} =  \begin{bmatrix} \lambda_1 \overrightarrow{\alpha ^T_1} & \lambda_2 \overrightarrow{\alpha ^T_2} & ...& \lambda_n \overrightarrow{\alpha ^T_n} \end{bmatrix}  \\
&\Rightarrow	\qquad	P\alpha^T_i = \lambda_i \alpha^T_i \qquad i = 1,2,...,n

\end{aligned}
$$





## 矩阵的特征向量(eigenvector)、特征值(eigenvalue)

$$
A\overrightarrow{\alpha} = \lambda\overrightarrow{\alpha}
$$
*	本质
	*	方阵$A$代表着从一个向量空间到另一个向量空间的线性映射，特征向量$\alpha$在此映射下没有发生旋转，只是发生了伸缩，缩放因子就是其对应的特征值$\lambda$
	*	在一定条件下（如$A$为实对称矩阵时），一个线性映射可以由其特征值和特征向量完全表述，也就是说：所有的特征向量组成了这向量空间的一组基底。



## 矩阵的奇异向量、奇异值
$$
M\overrightarrow{v} = \sigma \overrightarrow{u}\\
M^*\overrightarrow{u} = \sigma \overrightarrow{v}
$$
*	$M$是域$K$上的$m\times n$阶矩阵。
*	满足上两式非负实数$\sigma$是矩阵M的一个奇异值，单位向量$\overrightarrow{u}, \overrightarrow{v}$分别是$\sigma$的左奇异向量、右奇异向量。
*	一个$m\times n$的矩阵至多有$p = min(m, n)$个不同的奇异值；
*	总能在$K^m$中找到由$M$的左奇异向量组成的一组正交基$U$，；
*	总能在$K^m$找到由$M$的右奇异向量组成的一组正交基$V$，










***
***
***
# 特殊的矩阵
## 初等矩阵

## 正交矩阵(orthogonal matrix)
*	定义：方块矩阵，元素为实数，行向量和列向量都两两正交。
*	性质：
	*	$$	A^T = A^{-1}$$	
	*	$$	det(A) = \pm1$$
	*	$$A$$的行（列）向量组是欧几里得空间$$R^n$$上的一个标准正交基。
	*	正交矩阵表示的线性映射（正交变换）是一个保距映射，即变换后向量长度不变。
		*	旋转、反射都是正交变换。
$$
R(\theta) = \begin{bmatrix}cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}
$$
*	群论相关
  *	所有的$$n\times n$$正交矩阵构成了一个正交群$$SO(n)$$，正交矩阵和正交矩阵的乘积依然是正交矩阵。



## 实对称矩阵(real symmetric matrix)



## 酉矩阵













***
***
***
# 线性空间、线性映射
## 向量空间（线性空间）
*	给定域F，F上的向量空间V是一个集合，其上定义了两种二元运算：
	*	向量加法 + : V × V → V，把V中的两个元素$\overrightarrow{u}$和$\overrightarrow{v}$映射到V中另一个元素，记作$\overrightarrow{u} + \overrightarrow{v}$；
	*	标量乘法 · : F × V → V，把F中的一个元素 a 和 V 中的一个元素$\overrightarrow{u}$变为V中的另一个元素，记作$a\cdot\overrightarrow{u}$。
*	在现代数学中，满足上面公理的任何数学对象都可被当作向量处理。
  *	譬如，实系数多项式的集合在定义适当的运算后构成向量空间，在代数上处理是方便的。
  *	单变元实函数的集合在定义适当的运算后，也构成向量空间，研究此类函数向量空间的数学分支称为泛函分析。	



## 欧几里得空间(Euclidean space)
*	定义：$$n$$维向量空间$$R^n$$，加上内积定义，就是一个欧几里得空间




## 基(basis)
*	是向量空间的一个特殊的子集。
	*	基的元素称为基向量。
	*	向量空间中的任意一个元素，都可以表示成基向量的线性组合。
	*	有限维向量空间中，基向量的个数就是向量空间的维数。	
*	**标准正交基(standard orthogonal basis)**
	*	定义了内基的向量空间（欧几里得空间）中，两两正交的单位基向量构成的基。



## 线性映射(linear mapping)
*	是在两个向量空间（包括由函数构成的抽象的向量空间）之间的一种保持向量加法和标量乘法的特殊映射
*	如果向量空间V和W是有限维的，并且在这些空间中有选择好的基，则从V到W的所有线性映射可以被表示为矩阵



## 算子(operator)

*	是从一个向量空间（或模）到另一个向量空间（或模）的映射。
*	设$U$、$V$是两个向量空间。从$U$到$V$的任意映射被称为算子。



##	零空间(null space)

*	一个算子$A$的零空间是方程$A\overrightarrow{x} = 0$的所有解$\overrightarrow{\alpha}$的集合。
*	算子$A$的零空间也叫做$A$的核, 核空间，记做$kerA$。
*	如果$A$是矩阵，它的零空间就是所有向量的空间的线性子空间。
	*	矩阵$A$的零空间的维度叫做$A$的零化度(nullity)。
	*	矩阵$A$的零空间是齐次线性方程组$A\overrightarrow{x} = 0$的解。





***
***
***
# 多项式环
## 二次型

 


***
***
***
# 线性代数中的算法
## Gauss消元法(Gaussian Elimination)
*	求矩阵的秩一个算法。
*	进一步可以利用其求线性方程组的解，以及求可逆矩阵的逆。
*	思路：使用初等行变换，将矩阵化为行阶梯型矩阵(Row Echelon Form matrix)
*	算法复杂度是O(n3)


## Gauss-Jordan消元法(Gauss-Jordan Elimination)
*	思路：在Gauss消元法的基础上进一步进行初等行变换，将矩阵化为简化行阶梯
型矩阵(reduced row echelon form matrix)。


## LU分解(LU decomposition)
*	方阵$A$的LU分解：将方阵分解成一个下三角矩阵(lower triangular matrix)$L$和一个
上三角矩阵(Upper triangular matrix)$U$的乘积：$A=LU$
*	在数值计算上，LU分解经常被用来解线性方程组，且在求逆矩阵和计算行列式中都是
一个关键的步骤。


## PLU分解(PLU decomposition)
*	将方阵A分解成一个置换矩阵$P$、下三角矩阵$L$、上三角矩阵$U$



## 奇异值分解(SVD: singular value decomposition)

