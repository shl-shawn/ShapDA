

# ShapDA: Interpretable Domain Adaptation Enables Robust Lactic Acid Fermentation Monitoring from Waste

**Unsupervised Domain Adaptation in Lactic Acid Production**: The source domain uses simple glucose (SG) while the target domain uses complex bio-waste hydrolysate (CW). Subplot (a) shows FTIR data for the source ($X_s$) and target ($X_t$), highlighting the domain shift. Subplot (b) presents ShapDA, which identifies domain-invariant features. A regression model trained on these features enhances target predictions ($\hat{y}_t$), mitigating the effects of domain shift.

<img src="./assets/grabs.jpg" alt="alt text" width="200%" height="150%">


<a name="data"></a>
## 🧪 Data
The dataset is available at [Zenodo](https://doi.org/10.5281/zenodo.14171429). The files should be extracted and placed in the `dataset` folder. 


<a name="installation"></a>
## ⚙️ Installation

To reproduce our results, please kindly create and use the following environment.

```python
git clone https://github.com/shl-shawn/ShapDA.git
cd ShapDA
conda create -n ShapDA python
conda activate ShapDA
pip install -r requirements.txt
```



<a name="training-and-evaluation"></a>
## 🤖 Training and Evaluation

The following deep models can be executed with their default parameters. To run the training and testing scripts, ensure that the correct paths to the dataset, model weights, and save directory are specified (i.e., `weight_path`, `dataset_dir` and `save_dir`).


### Domain Adversarial Neural Networks for Regression (DANN-R)
```python
#Train
cd src/DANN-R
python train.py

#Test
cd src/DANN-R
python test.py
```

### Domain Adaptation Regression with GRAM matrices (DARE-GRAM)
```python
#Train
cd src/DARE-GRAM
python train.py

#Test
cd src/DARE-GRAM
python test.py
```

### Deep Correlation Alignment for Regression (DeepCORAL-R)
```python
#Train
cd src/DeepCORAL-R
python train.py

#Test
cd src/DeepCORAL-R
python test.py
```

### ShapDA

```bash
cd src/ShapDA
```
- **`shapfs_glucose.ipynb`**: Implements the ShapDA method for predicting glucose concentration in both source and target domains.
- **`shapfs_lacticacid.ipynb`**: Applies ShapDA to predict lactic acid concentration across source and target domains.
- **`shapfs_combine.ipynb`**: Combines domain-invariant features for glucose and lactic acid to evaluate their impact on prediction accuracy.


<a name="results"></a>
## 📊 Results

### Glucose prediction

<table>
  <thead>
    <tr>
      <th rowspan="2">DA Methods</th>
      <th colspan="4">Source (SG)</th>
      <th colspan="4">Target (CW)</th>
    </tr>
    <tr>
      <th>R² ↑</th>
      <th>RMSE [g/L] ↓</th>
      <th>RMSEP [%] ↓</th>
      <th>RPD ↑</th>
      <th>R² ↑</th>
      <th>RMSE [g/L] ↓</th>
      <th>RMSEP [%] ↓</th>
      <th>RPD ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap;">No Adaptation</td>
      <td><strong>0.98</strong></td>
      <td>4.4</td>
      <td>3.8</td>
      <td>8.3</td>
      <td>0.56</td>
      <td>20.4</td>
      <td>14.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">DeepCORAL-R</td>
      <td>0.80</td>
      <td>18.0</td>
      <td>15.9</td>
      <td>2.2</td>
      <td>0.69</td>
      <td>17.3</td>
      <td>12.0</td>
      <td>1.8</td>
    </tr>
    <tr>
      <td>DANN-R</td>
      <td>0.89</td>
      <td>13.0</td>
      <td>11.5</td>
      <td>3.1</td>
      <td>0.86</td>
      <td>11.5</td>
      <td>8.0</td>
      <td>2.7</td>
    </tr>
    <tr>
      <td>DARE-GRAM</td>
      <td><strong>0.98</strong></td>
      <td>5.0</td>
      <td>4.5</td>
      <td>8.0</td>
      <td>0.92</td>
      <td>8.7</td>
      <td>6.0</td>
      <td>3.6</td>
    </tr>
    <tr style="background-color: #d1e7dd;">
      <td><strong>ShapDA (ours)</strong></td>
      <td><strong>0.98</strong></td>
      <td><strong>4.3</strong></td>
      <td><strong>3.7</strong></td>
      <td><strong>8.5</strong></td>
      <td><strong>0.96</strong></td>
      <td><strong>5.8</strong></td>
      <td><strong>4.0</strong></td>
      <td><strong>5.4</strong></td>
    </tr>
  </tbody>
</table>

*Table 1: Unsupervised domain adaptation results for glucose prediction in source (SG) and target (CW) fermentations. Arrows indicate desired direction: ↑ (higher is better), ↓ (lower is better).*


### Lactic acid prediction

<table>
  <thead>
    <tr>
      <th rowspan="2">DA Methods</th>
      <th colspan="4">Source (SG)</th>
      <th colspan="4">Target (CW)</th>
    </tr>
    <tr>
      <th>R² ↑</th>
      <th>RMSE [g/L] ↓</th>
      <th>RMSEP [%] ↓</th>
      <th>RPD ↑</th>
      <th>R² ↑</th>
      <th>RMSE [g/L] ↓</th>
      <th>RMSEP [%] ↓</th>
      <th>RPD ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap;">No Adaptation</td>
      <td><strong>0.99</strong></td>
      <td>3.3</td>
      <td>3.9</td>
      <td>9.0</td>
      <td>0.79</td>
      <td>9.4</td>
      <td>10.6</td>
      <td>2.2</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">DeepCORAL-R</td>
      <td>0.95</td>
      <td>6.1</td>
      <td>6.9</td>
      <td>4.6</td>
      <td>0.82</td>
      <td>8.4</td>
      <td>9.5</td>
      <td>2.4</td>
    </tr>
    <tr>
      <td>DANN-R</td>
      <td>0.88</td>
      <td>9.5</td>
      <td>10.8</td>
      <td>3.9</td>
      <td>0.83</td>
      <td>8.2</td>
      <td>9.2</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>DARE-GRAM</td>
      <td>0.90</td>
      <td>8.9</td>
      <td>10.2</td>
      <td>3.1</td>
      <td>0.87</td>
      <td>7.3</td>
      <td>8.2</td>
      <td>2.8</td>
    </tr>
    <tr style="background-color: #d1e7dd;">
      <td><strong>ShapDA (ours)</strong></td>
      <td><strong>0.99</strong></td>
      <td><strong>3.1</strong></td>
      <td><strong>3.7</strong></td>
      <td><strong>9.7</strong></td>
      <td><strong>0.91</strong></td>
      <td><strong>6.1</strong></td>
      <td><strong>6.8</strong></td>
      <td><strong>3.3</strong></td>
    </tr>
  </tbody>
</table>

*Table 2: Unsupervised domain adaptation results for lactic acid prediction from source (SG) to target (CW) fermentations. Arrows indicate desired direction: ↑ (higher is better), ↓ (lower is better).*

<a name="citation"></a>
## 📚 Citation
If you use our work or dataset in your work, please cite it as:

### APA
```
Babor, M., Liu, S., Arefi, A., Olszewska-Widdrat, A., Sturm, B., Venus, J., & Höhne, M. M. (2025). Interpretable Domain Adaptation Enables Robust Lactic Acid Fermentation Monitoring from Waste. Results in Engineering. [https://doi.org/10.2139/ssrn.5012080](https://doi.org/10.1016/j.rineng.2025.108477)
```

### BibTex
```
@article{BABOR2025108477,
title = {Interpretable Domain Adaptation Enables Robust Lactic Acid Fermentation Monitoring from Waste},
journal = {Results in Engineering},
pages = {108477},
year = {2025},
issn = {2590-1230},
doi = {https://doi.org/10.1016/j.rineng.2025.108477},
url = {https://www.sciencedirect.com/science/article/pii/S2590123025045219},
author = {Majharulislam Babor and Shanghua Liu and Arman Arefi and Agata Olszewska-Widdrat and Joachim Venus and Barbara Sturm and Marina M.C. Höhne},
}
```

## 🙏 Acknowledgement 
+ DANN-R is developed from [DANN](https://github.com/NaJaeMin92/pytorch-DANN).  
+ DARE-GRAM is used as our [codebase](https://github.com/ismailnejjar/DARE-GRAM). 
+ DeepCORAL-R is developed from [DeepCORAL](https://github.com/SSARCandy/DeepCORAL).
