---
layout: post
comments: true
title:  "MathJax Examples"
date:   2030-06-12 22:00:00
categories: math
---


$$ \frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h} $$

$$ f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x $$

$$ \frac{mc^2}{\sqrt{1-\frac{v^2}{c^2}}} $$

$$ \frac{\partial E}{\partial u_j} = y_j - t_j $$

$$ \frac{1}{2} \sum_{j=1}^n\left(y_j - t_j \right)^2 $$

\\( \Sigma_C = \sum_{d=1}^C e^{z_d} \\)

$$ \Sigma_C = \sum_{d=1}^C e^{z_d} $$

$$ f(x_i, W, b) = W x_i + b $$

$$ L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta) $$

$$ L_i = \max(0, -7 - 13 + 10) + \max(0, 11 - 13 + 10) $$

$$ L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta) $$

$$ L = \underbrace{ \frac{1}{N} \sum_i L_i }\text{data loss} + \underbrace{ \lambda R(W) }\text{regularization loss} \\ $$

$$ L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)j - f(x_i; W){y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2 $$

$$ L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j} $$

$$ H(p,q) = - \sum_x p(x) \log q(x) $$

$$ P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} } $$


$$L_j = E_j$$

$$a^2 + b^2 = c^2$$

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

$$ \mathsf{Data = PCs} \times \mathsf{Loadings} $$

\\[ \mathbf{X} = \mathbf{Z} \mathbf{P^\mathsf{T}} \\]

$$ \mathbf{X}\_{n,p} = \mathbf{A}\_{n,k} \mathbf{B}\_{k,p} $$

\\( sin(x^2) \\)
