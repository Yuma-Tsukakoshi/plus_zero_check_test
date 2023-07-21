# 混同行列とは  
機械学習で分類モデルの性能を評価するために使用される表  
真陽性 (True Positive, TP)：実際に陽性のデータ（Positive）を正しく陽性として分類した数。  
偽陰性 (False Negative, FN)：実際に陽性のデータ（Positive）を誤って陰性として分類した数。  
偽陽性 (False Positive, FP)：実際に陰性のデータ（Negative）を誤って陽性として分類した数。  
真陰性 (True Negative, TN)：実際に陰性のデータ（Negative）を正しく陰性として分類した数。  
これらの4つの要素を用いて混同行列を表現する。  

# 混同行列表記
|       |        |            |Predict |
|:------|:------:|:------:|-------:|
|       |        |Positive|Negative|
|truth  |Positive|TP      |FN      |
|^      |Negative|FP      |TN      |

上記の混同行列をもとに、以下にPrecision,Recall,F値,IoUの式と意味を示す。
## Precision
Precisionは、適合率のことを指し、Positiveと予測したクラスの内実際にPositiveであった割合  
式で表すと以下のようになる。
``` math
Precision = \frac{TP}{TP+FP}

```
## Recall
Recallは、再現率のことを指し、実際のPositiveクラスのうち正しくPositiveと予想出来た割合
式で表すと以下のようになる。  
``` math
Recall = \frac{TP}{TP+FN}

```

## F値
F値は、適合率(PRE)と再現率(REC)の調和平均を指し、適合率と再現率が共に高い場合に、F1 スコアも高くなる。  
式で表すと以下のようになる。  
``` math
F1 = \frac{2 \times (Precision \times Recall)}{Precision + Recall}

```
## IoU
IoUは、Intersection over Unionの略で、2つの領域がどれだけ重なっているかを表す指標。  
IoUは物体検出の分野における評価指標として使われる。  
式で表すと以下のようになる。  
``` math
IoU = \frac{TP}{TP + FP + FN}

```




