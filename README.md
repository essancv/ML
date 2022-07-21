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

Training Set ==> lerning Algorithm ==> function (model)

any x => function (model) => predict  $\hat{y}$

Used in:
<ul>
  <li> Regression problems (i.e:predict a continuous value). Can be linear or logistic regression </li>
    <ul>
      <li> Linear regression with one variable  </li>
      <ul>
        <li>Model ==> $f_{w,b}= wx+ b$ (or f(x) = wx + b </li>
        <li> Cost Function (Squeared Error cost Function) ==> $J_{(w,n)}=\frac{1}{2m}\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)} )^2$ </li>
        <li> $\hat{y}^{(i)}=f_{w,b}(x^{(i)})$</li>
        <li> $f_{w,b}(x^{(i)})=wx^{(i)}+ b$</li>
        <li> Goal: Buscar w,b que haga que la predicción sea más aproximada al valor real para cada muestra del set de prueba</li>
      </ul>
    </ul>        
  <li> Classification problems (predict categories, discrete values outputs, i.e:yes/not , benign/malignant/ can have more than two outputs)
  <ul>
    <li> Logistic Regression </li>
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
