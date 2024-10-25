# ShapFS
Code for our paper: Domain-Invariant Monitoring for Lactic Acid Production: Transfer Learning from Glucose to Bio-Waste Using Machine Learning Interpretation


<img src="./assest/Figure_1.jpg" alt="alt text" width="200%" height="150%">

## Installation
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.

```python
git clone https://github.com/shl-shawn/ShapFS.git
cd ShapFS
conda create -n ShapFS python
conda activate dann
pip install -r requirements.txt
```

## Train and Test Deep Learning models
The program can be executed with the default parameters. To run the training and testing scripts, ensure that the correct paths to the dataset, model weights, and save directory are specified (i.e., `weight_path`, `dataset_dir` and `save_dir`).

### DANN-R
```python
#Train
cd src/DANN-R
python train.py

#Test
cd src/DANN-R
python test.py
```

### DARE-GRAM
```python
#Train
cd src/DARE-GRAM
python train.py

#Test
cd src/DARE-GRAM
python test.py
```

### DeepCORAL-R
```python
#Train
cd src/DeepCORAL-R
python train.py

#Test
cd src/DeepCORAL-R
python test.py
```


## Experiment results

### Glucose prediction experiment results

| **DA Methods**   | **Source (SG)**     |       |       |       |       | **Target (CW)**    |       |       |       |
|------------------|:------------------:|:-----:|:-----:|:-----:|:-----:|:-----------------:|:-----:|:-----:|:-----:|
|                  | **R² ↑**           | **RMSE [g/L] ↓**   | **RMSEP [%] ↓** | **RPD ↑** |       | **R² ↑**        | **RMSE [g/L] ↓** | **RMSEP [%] ↓** | **RPD ↑** |
| No Adaptation| **0.98**  | 4.4                | 3.8       | 8.3       |  | 0.56      | 20.4              | 14.2       | 1.5       |
| DeepCORAL-R  | 0.80      | 18.0               | 15.9      | 2.2       |  | 0.69      | 17.3              | 12.0       | 1.8       |
| DANN-R       | 0.89      | 13.0               | 11.5      | 3.1       |  | 0.86      | 11.5              | 8.0        | 2.7       |
| DARE-GRAM    | **0.98**  | 5.0                | 4.5       | 8.0       |  | 0.92      | 8.7               | 6.0        | 3.6       |
| **ShapFS**       | **0.98**  | **4.3**            | **3.7**   | **8.5**   |  | **0.96**  | **5.8**           | **4.0**    | **5.4**   |

*Table 1: Unsupervised domain adaptation results for glucose in source fermentation using glucose as the substrate (SG) and target fermentation using complex sugar from waste as the substrate (CW). Here DeepCORAL-R is Deep Correlation Alignment for Regression, DANN-R is Domain Adversarial Neural Networks for Regression, and ShapFS is SHapley Additive exPlanations-based domain invariant feature selection method.*


### Lactic acid prediction experiment results

| **DA Methods**   | **Source (SG)**     |       |       |       |       | **Target (CW)**    |       |       |       |
|------------------|:------------------:|:-----:|:-----:|:-----:|:-----:|:-----------------:|:-----:|:-----:|:-----:|
|                  | **R² ↑**           | **RMSE [g/L] ↓**   | **RMSEP [%] ↓** | **RPD ↑** |       | **R² ↑**        | **RMSE [g/L] ↓** | **RMSEP [%] ↓** | **RPD ↑** |
| No Adaptation| **0.99**           | 3.3   | 3.9   | 9.0   |       | 0.79            | 9.4   | 10.6  | 2.2   |
| DeepCORAL-R  | 0.95               | 6.1   | 6.9   | 4.6   |       | 0.82            | 8.4   | 9.5   | 2.4   |
| DANN-R       | 0.88               | 9.5   | 10.8  | 3.9   |       | 0.83            | 8.2   | 9.2   | 2.5   |
| DARE-GRAM    | 0.90               | 8.9   | 10.2  | 3.1   |       | 0.87            | 7.3   | 8.2   | 2.8   |
| **ShapFS**       | **0.99**           | **3.1** | **3.7** | **9.7** |       | **0.91**        | **6.1** | **6.8** | **3.3** |

*Table 2: Lactic acid prediction results using unsupervised domain adaptation from source fermentation using glucose as the substrate (SG) to target fermentation using complex sugar from waste as the substrate (CW).*
 


## Data
The files should be extracted and placed in the `dataset` folder. For access to the dataset, please contact the corresponding author.


## Acknowledgement 
+ DANN-R is developed from DANN [official](https://github.com/NaJaeMin92/pytorch-DANN)  
+ DARE-GRAM is used as our codebase [official](https://github.com/ismailnejjar/DARE-GRAM)  
+ DeepCORAL-R is developed from DeepCORAL [official](https://github.com/SSARCandy/DeepCORAL) 


## Contact
For questions regarding the code and data, please contact MBabor@atb-potsdam.de or SLiu@atb-potsdam.de .
