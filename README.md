
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![tkinter](https://img.shields.io/badge/tkinter-Latest-blue?style=flat&logo=tkinter)](https://tkinter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-blueviolet?style=flat&logo=numpy)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.3.1-orange)](https://pandas.pydata.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.11.2-blue)](https://seaborn.pydata.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-brightgreen)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-red)](https://scikit-learn.org/)
[![tkinter](https://img.shields.io/badge/tkinter-8.6-yellow)](https://docs.python.org/3/library/tkinter.html)


</div>

<h1 align="center"> Cross-Validation With GUI</h1>

## Table of Contents
* [General Info](#general-info)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Demo](#demo)
* [Contributing](#contributing)


## General Info ℹ️
The "Cross Validation with GUI" project aims to develop a user-friendly graphical interface for conducting cross-validation, a fundamental technique in machine learning model evaluation. Cross-validation is crucial for assessing a model's performance and generalizability by partitioning the dataset into multiple subsets and iteratively training and testing the model.

## Technologies Used 💻
This project is created using:
- **Python** (`Python`) : Any version above 3.8
- **numpy** (`numpy`): For numerical operations and array manipulations.
- **pandas** (`pandas`): For data manipulation and analysis using DataFrame structures.
- **matplotlib** (`matplotlib`): For data visualization, including plotting graphs and charts.
- **scikit-learn** (`sklearn`): For machine learning algorithms and tools.
  - `datasets`: Provides built-in datasets for experimentation.
  - `model_selection`: Includes functions for model selection and evaluation, such as cross-validation.
  - `neighbors` (`KNeighborsClassifier`): Implements the k-nearest neighbors classification algorithm.
  - `mixture` (`GaussianMixture`): Implements Gaussian Mixture Models for clustering.
  - `preprocessing` (`StandardScaler`): Provides tools for data preprocessing, such as standardization.
  - `svm` (`SVC`): Implements Support Vector Machine (SVM) algorithms for classification.
  - `metrics` (`confusion_matrix`): Includes evaluation metrics like confusion matrix.
- **seaborn** (`seaborn`): Enhances the visualization of statistical data using Matplotlib.
- **tkinter** (`tkinter`): Provides GUI functionality for Python applications.
  - `filedialog`: Allows opening file dialogs to interact with files and directories.
  - `ttk`: Provides themed widgets for GUI development.
- **matplotlib.backends.backend_tkagg** (`FigureCanvasTkAgg`): Enables embedding Matplotlib figures into Tkinter applications.


## Features ✨
* Supports various cross-validation methods including K-Nearest Neighbours, Gaussian Mixture Model,Support Vector Classification.
* Graphical User Interface (GUI) for interactive model evaluation and comparison.
* Choose from three different datasets directly through the GUI for seamless validation. (Iris, Wine, Breast)
* Visualize model performance metrics for each fold or iteration.

## Setup 🛠️
To run this project locally, follow these steps:

1. Install tkinter using pip:
```bash
$ pip install tkinter
```
2. Install Numpy:
```bash
$ pip install numpy
```
3. Install Pandas:
```bash
$ pip install pandas
```
4. Install scikit:
```bash
$ pip install scikit-learn
```
5. Install matplitlib:
```bash
$ pip install matplotlib
```
## Usage ▶️
Once the project is set up, you can run the main script to start the Cross-Validation GUI application. Use the interface to load your dataset, choose machine learning models, and select cross-validation settings. The GUI will guide you through the process of training and evaluating your models.😊

## Demo 🎬
<div align="center">

   ![assets/Screenshot 1.png](https://github.com/kashifmehdi/Air-Canva/blob/8d9ffb4995101ccf1808b976b5b4c005ff5011ad/assets/Screenshot%201.png)
   <p>Color Palette Screenshot </p>
   
   ![assets/Screenshot 2.png](https://github.com/kashifmehdi/Air-Canva/blob/8d9ffb4995101ccf1808b976b5b4c005ff5011ad/assets/Screenshot%202.png)
   <p>Canvas in action with diffrent brush sizes </p>
</div>


## Contributing 🤝
Contributions are welcome! If you'd like to contribute to this project, please open an issue or submit a pull request on GitHub.

## Contact Information 
For questions, feedback, or collaboration opportunities, feel free to contact the project maintainer at [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:your-email@example.com) or [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/your-profile/).
