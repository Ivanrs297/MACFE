# MACFE
## Meta-learning and Causality Based Feature Engineering

Paper: https://link.springer.com/chapter/10.1007/978-3-031-19493-1_5

### Instructions
1. Original datasets to transform should be on `datasets_input/` folder. There is inside an example of dataset (*sonar*).
2. To make use of MACFE in a GridSearch fashion, run the file `run.py`.
3. Output datasets transformed are on `datasets_output/` folder
4. A file `results.csv` will be generated with the evaluation results of F1-score, Accuracy and AUC for eight classifiers (KNN, LR, SVC-L, SVC-P, RF, AB, MLP, DT)


Cite this work:

`
@InProceedings{10.1007/978-3-031-19493-1_5,
author="Reyes-Amezcua, Ivan
and Flores-Araiza, Daniel
and Ochoa-Ruiz, Gilberto
and Mendez-Vazquez, Andres
and Rodriguez-Tello, Eduardo",
editor="Pichardo Lagunas, Obdulia
and Mart{\'i}nez-Miranda, Juan
and Mart{\'i}nez Seis, Bella",
title="MACFE: A Meta-learning and Causality Based Feature Engineering Framework",
booktitle="Advances in Computational Intelligence",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="52--65",
abstract="Feature engineering has become one of the most important steps to improving model prediction performance, and producing quality datasets. However, this process requires non-trivial domain knowledge which involves a time-consuming task. Thereby, automating such processes has become an active area of research and interest in industrial applications. In this paper, a novel method, called Meta-learning and Causality Based Feature Engineering (MACFE), is proposed; our method is based on the use of meta-learning, feature distribution encoding, and causality feature selection. In MACFE, meta-learning is used to find the best transformations, then the search is accelerated by pre-selecting ``original'' features given their causal relevance. Experimental evaluations on popular classification datasets show that MACFE can improve the prediction performance across eight classifiers, outperforms the current state-of-the-art methods on average by at least 6.54{\%}, and obtains an improvement of 2.71{\%} over the best previous works.",
isbn="978-3-031-19493-1"
}
`

