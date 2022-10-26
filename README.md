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
author="Reyes-Amezcua, Ivan",
title="MACFE: A Meta-learning and Causality Based Feature Engineering Framework",
booktitle="Advances in Computational Intelligence",
year="2022",
publisher="Springer Nature Switzerland",
pages="52--65",
isbn="978-3-031-19493-1"
}
`

