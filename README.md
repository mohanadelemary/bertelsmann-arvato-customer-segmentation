# Arvato Bertelsmann Customer Segmentation Project

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Libraries Used](#libraries-used)
3. [Project Files](#project-files)
4. [Analysis and Results](#analysis-and-results)
5. [Conclusion](#conclusion)
6. [Acknowledgements](#acknowledgements)

---

### Project Motivation
This project, based on the Arvato Bertelsmann Kaggle dataset, aims to analyze customer demographics to identify patterns that can help predict which segments of the general population are likely to respond to a marketing campaign from a German mail-order company. This involves a two-part approach:
1. **Customer Segmentation**: Understanding different customer profiles through clustering and demographic analysis.
2. **Predictive Modeling**: Using supervised learning to predict potential customers from a general population dataset.

By successfully segmenting customers and building a robust model, the project seeks to provide actionable insights for targeted marketing and help improve the campaign's return on investment.

### Libraries Used
The following libraries were essential in developing and analyzing this project:

- **Requests**: For handling HTTP requests, particularly to download data files.
- **Tarfile** and **OS**: For file extraction and file path management.
- **Time**: For tracking code execution time.
- **Pandas** and **NumPy**: For data manipulation, analysis, and preprocessing.
- **Matplotlib** and **Seaborn**: For data visualization to explore data distributions and relationships.
- **Scipy**: 
  - **stats**: For statistical tests, including `ttest_ind` and `chi2_contingency`.
- **Scikit-learn**:
  - **KNNImputer**: For imputing missing values.
  - **OneHotEncoder**: For encoding categorical features.
  - **PCA**: For dimensionality reduction to handle high-dimensional data.
  - **StandardScaler**: For feature scaling to improve model performance.
  - **KMeans**: For clustering in the customer segmentation step.
  - **Train_test_split** and **StratifiedKFold**: For splitting data into training/testing sets and for stratified cross-validation to handle class imbalance.
  - **Learning_curve**: To plot learning curves and evaluate model performance over varying sample sizes.
  - **GridSearchCV** and **BayesSearchCV** (via `skopt`): For hyperparameter optimization, with `BayesSearchCV` providing a Bayesian optimization alternative to `GridSearchCV`.
  - **GradientBoostingClassifier** and **AdaBoostClassifier**: For ensemble-based classification models.
  - **RandomForestClassifier**: For building an alternative classification model.
  - **Metrics**: For model evaluation, including `accuracy_score`, `recall_score`, `f1_score`, `roc_auc_score`, `precision_score`, `classification_report`, `roc_curve`, and `precision_recall_curve`.
- **Imbalanced-learn**:
  - **SMOTE**: For handling class imbalance by oversampling the minority class.
- **Scikit-optimize (`skopt`)**:
  - **BayesSearchCV** and **space (Real, Integer)**: For Bayesian optimization in hyperparameter tuning.


### Project Files

- **README.md**: This file, providing an overview of the project.
- **DIAS Attributes - Values 2017.xlsx**: This Excel file contains the attributes and corresponding values for the DIAS dataset, specifically for the year 2017. 
                                        It is used for analyzing the different characteristics and values associated with the dataset.
- **DIAS Information Levels - Attributes 2017.xlsx**: This Excel file provides information on the attributes and their respective levels for the DIAS dataset in 2017. 
                                        It serves as a reference for understanding the data structure and categorization used in the project.
- **Arvato_Project_Workbook.ipynb**: The primary Jupyter Notebook with all the code for data exploration, cleaning, customer segmentation, clustering analysis and the code for training and evaluating the supervised learning model
- **results.csv**: Folder containing saved models and evaluation results from the best performing models.

### Analysis and Results
1. **Data Cleaning & Preprocessing**:
   - Analyzed and handled large amounts of missing data, particularly in demographics.
   - Conducted feature engineering, one-hot encoding, and standardization to prepare data for modeling.
   - Applied PCA for dimensionality reduction to address high feature dimensionality.

2. **Customer Segmentation**:
   - Conducted clustering analysis to segment customers into meaningful groups.
   - Identified specific clusters of customers to focus future marketing efforts.

3. **Predictive Modeling**:
   - Built a binary classification model to predict whether an individual from the general population would likely become a customer.
   - Used a Gradient Boosting Classifier and optimized it using Bayesian search to maximize ROC AUC.
   - Final model achieved a ROC AUC score of approximately 0.77, demonstrating a good ability to distinguish potential customers.

### Conclusion
This project showcases the value of customer segmentation and predictive modeling in marketing strategy. By clustering customers, we uncovered key demographics that characterize high-value segments, while the predictive model provides an effective way to identify prospective customers within the general population. These insights can drive better-targeted marketing efforts and improve campaign ROI.

### Acknowledgements
Special thanks to:
- **Arvato Bertelsmann** and **Udacity** for providing the data and project context.
- The **Kaggle community** for discussions and resources related to the Arvato dataset.
- Various StackOverflow and Kaggle contributors for guidance on handling data preprocessing challenges.

### Results

Results recap and full blog article can be found [here](https://medium.com/@elemary.mohanad/customer-segmentation-and-predictive-modeling-for-marketing-campaigns-a-case-study-with-arvato-d38e612b6afb) 
