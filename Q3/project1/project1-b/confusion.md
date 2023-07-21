# 混同行列とは  
機械学習で分類モデルの性能を評価するために使用される表  
真陽性 (True Positive, TP)：実際に陽性のデータ（Positive）を正しく陽性として分類した数。  
偽陰性 (False Negative, FN)：実際に陽性のデータ（Positive）を誤って陰性として分類した数。  
偽陽性 (False Positive, FP)：実際に陰性のデータ（Negative）を誤って陽性として分類した数。  
真陰性 (True Negative, TN)：実際に陰性のデータ（Negative）を正しく陰性として分類した数。  
これらの4つの要素を用いて混同行列を表現する。  

# 混同行列表記
|>      |        |>       |Predict |
|^      |        |Positive|Negative|
|:------|:------:|:------:|-------:|
|truth  |Positive|TP      |FN      |
|^      |Negative|FP      |TN      |

上記の混同行列をもとに、以下にPrecision,Recall,F値,IoUの式と意味を示す。
