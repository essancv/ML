# Machine Learning (ML)
ML examples:
<ul> 
  <li> Autonomous Driving </li>
  <li> Natural Language Processing (NLP) </li>
  <li> Computer vision </li>
  <li> Product Recommendations </li>
</ul>

## Supervised Learning 

Algoritmos que aprenden cuando se les entrena con "respuestas correctas" X(features) --> Y (target)

Uses: Spam filtering, speech recognition, machine translation, online advertising,self-driving car,visual inspection

Training Set ==> lerning Algorithm (Gradient Descent) ==> function (model)

any x => function (model) => predict  $\hat{y}$

Used in:
<ul>
  <li> Regression problems (i.e:predict a continuous value). Can be linear or logistic regression </li>
    <ul>
      <li> Linear regression with one feature  </li>
      <ul>
        <li>Model ==> $f_{w,b}= wx+ b$ (or f(x) = wx + b </li>
        <li> Cost Function (Squeared Error cost Function) ==> $J_{(w,b)}=\frac{1}{2m}\sum{_{i=1}^m}(\hat{y}^{(i)}-y^{(i)} )^2$ </li>
        <li> $\hat{y}^{(i)}=f_{w,b}(x^{(i)})$</li>
        <li> $f_{w,b}(x^{(i)})=wx^{(i)}+ b$</li>
        <li> Goal: Buscar w,b que haga que la función de coste sea lo más pequeán posible ( cada predicción sea más aproximada al valor real para cada muestra del set de prueba)</li>
        <li>  Gradient Descent :
          <ul>
            <li> Repeat until convergence</li>
            <li> $w = w - \alpha\frac{\partial}{\partial{w}}J(w,b)$ </li>
            <li> $b = b - \alpha\frac{\partial}{\partial{b}}J(w,b)$ </li>
            <li> $\alpha$  = Learning rate, normalmente valor entre (0 y 1 )$ </li>
          </ul>
         </li>
      </ul>
      <li> Linear regression with multiple features </li>
      <ul>
        <li>Model ==> $f_{w,b}= w_1x_1 + w_2x_2 + ..+ w_nx_n  b$ (or f_W(X) = W . X + b) , n=Nº features, W is a row vector $[w_1 w_2 ...w_n]$ y X is a row vector $[x_1,x_2...x_n]$ and . is a dot product. $x_j^{(i)}$ es el valor de la feacture j (de n) la i-esima fila del training set (de m muestras)</li>
      </ul>        
      <li>  Gradient Descent :
          <ul>
            <li> Repeat until convergence</li>
            <li> $w_1 = w_1 - \alpha\frac{\partial}{\partial{w_1}}J(W,b) = w_1 - \alpha\frac{1}{m}\sum{_{i=1}^{m}}(f_{(W,b)}(X^{(i)} - y^{(i)})x_{1}^{(i)})$ </li>
            <li> $w_n = w_n - \alpha\frac{\partial}{\partial{w_n}}J(W,b) = w_n - \alpha\frac{1}{m}\sum{_{i=1}^{m}}(f_{(W,b)}(X^{(i)} - y^{(i)})x_{n}^{(i)})$ </li>
            <li> $b = b - \alpha\frac{\partial}{\partial{b}}J(W,b) = b - \alpha\frac{1}{m}\sum{_{i=1}^{m}}(f_{(W,b)}(X^{(i)} - y^{(i)}))$ </li>
            <li> $\alpha$  = Learning rate, normalmente valor entre (0 y 1 )$ </li>
            <li>  Feature scaling : make GD converge more efficientely (when features have very different numeric scales)
            <ul>
              <li> $s_{1,scaled}= \frac{x_{1}}{Max x_{1}}$ </li>
              <li> Mean normalization $x_{1}=\frac{x_{1} - \mu_{1}}{Max x_{1} -  Min x_{1}}$ , where $\mu$ is average</li>
              <li> Z-score normalization $x_{1}=\frac{x_{1} - \sigma_{1}}{\sigma_{1}}$ , where $\sigma$ is standard deviation</li>
              </li>
            </ul>
            </li>
          </ul>
      </li>

      
  <li> Classification problems (predict categories, discrete values outputs, i.e:yes/not , benign/malignant/ can have more than two outputs)
  <ul>
    <li> Logistic Regression </li>
    <ul>
      <li> Sigmoid funtion => $g(z) = \frac{1}{1+e^{-z}}$ </li>
      <li> Model => $f_{w,b}= g(wx+ b)=\frac{1}{1+e^{-(wx+b)}}$ </li>
      <li> Cost funciont  $J(w,b)={\frac{1}{m}}{\sum{_{i=1}^{m}}}L(f_{w,b}(x^{(i)},y^{(i)})$ , where L is the loss function</li>
              <ul>
                <li> if $y^{/i)} = 1  , loss = -log(f_{w,b}(x^{(i)})$ </li>
                <li> if $y^{/i)} = 0  , loss = -log( 1 - f_{w,b}(x^{(i)})$ </li>
                <li> $L(f_{w,b}(x^{(i)},y^{(i)}) = -y^{(i)}log(f_{w,b}(x^{(i)}) - (1-y^{(i)})log( 1 - f_{w,b}(x^{(i)})$ </li>
                <li>  $J(w,b)=-{\frac{1}{m}}{\sum{_{i=1}^{m}}y^{(i)}log(f_{w,b}(x^{(i)}) + (1-y^{(i)})log( 1 - f_{w,b}(x^{(i)})}$ </li>
              </ul>
      <li> Gradient descent 
          <ul>
            <li> Repeat until convergence</li>
            <li> $w_j = w_j - \alpha\frac{\partial}{\partial{w_j}}J(W,b) = w_j - \alpha\frac{1}{m}\sum{_{i=1}^{m}}(f_{(W,b)}(X^{(i)} - y^{(i)})x_{j}^{(i)})$ </li>
            <li> $b = b - \alpha\frac{\partial}{\partial{b}}J(W,b) = b - \alpha\frac{1}{m}\sum{_{i=1}^{m}}(f_{(W,b)}(X^{(i)} - y^{(i)}))$ </li>
        </ul>
      </li>
    </ul>
    <li> Neural Networks </li>
    <ul>
      <li> High Bias (underfitting): More layers, more training , other NN architecture) </li>
      <li> Hign Variance (overfitting): More data, regularization (l2,dropout), other NN architecture) </li>
    </ul>
    <li> SVM (Support Vector Machines) </li>
  </ul>
    </li>
</ul>
  
## Unsupervised Learning
Los datos no están etiquetados y se busca alguna cosa 'especial' entre ellos (pattern)
Used in:
<ul>
  <li> Clustering data (i.e: market segmentation,social network analysis, organize computing clusters,google news..  </li>
  <li> Audio processing (SVD - Singular Value Decomposition , i.e: split voices that came from two different sources)</li>
  <li> PCA (Principal Component Analysis) (If there are many features we can reduce them using PCA)
  <li> Anomaly detection (fraud, manufacturing,monitoring computers in data center) </li>
</ul>
Algorithms:
<ul>
  <li> K-Means (Clustering)</li>
  <li> PCA  (Dimensionality reduction:compress data, data visualization)</li>
  <li> Anomaly Detection (fraud detection)</li>
</ul>

## Recommender Systems

## Reinforcement Learning
