
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![tkinter](https://img.shields.io/badge/tkinter-Latest-blue?style=flat&logo=tkinter)](https://tkinter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-blueviolet?style=flat&logo=numpy)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-Latest-orange)](https://pandas.pydata.org/)
[![seaborn](https://img.shields.io/badge/seaborn-Latest-blue)](https://seaborn.pydata.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-Latest-brightgreen)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-red)](https://scikit-learn.org/)
[![tkinter](https://img.shields.io/badge/tkinter-Latest-yellow)](https://docs.python.org/3/library/tkinter.html)


</div>

<h1 align="center"><b>Machine Learning Classifier Application</b></h1>

This is an enhanced machine learning classification application that demonstrates various classification algorithms on classic datasets. The application provides an interactive web interface built with Streamlit, making it easy to experiment with different classifiers and datasets.

## 🌟 Features

- Interactive web interface using Streamlit
- Support for multiple datasets (Examples used: Iris, Wine, Breast Cancer, Can be changed)
- Multiple classification algorithms:
  - K-Nearest Neighbors
  - Gaussian Mixture Model
  - Support Vector Classification
- Advanced visualization using Plotly
- Cross-validation scoring
- Confusion matrix visualization
- BIC score analysis for model selection
- Standardized data preprocessing
- Comprehensive performance metrics

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-classifier-app.git
cd ml-classifier-app
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Select a dataset
   - Choose a classifier
   - Run classification
   - Plot BIC scores

## 💡 Key Improvements from Original Version

1. **Modern Web Interface**: Replaced Tkinter with Streamlit for a more modern and responsive UI
2. **Enhanced Visualization**: Added interactive Plotly charts instead of static Matplotlib plots
3. **Code Organization**: Implemented proper class structure and type hints
4. **Error Handling**: Added robust error handling and user feedback
5. **Performance Metrics**: Added detailed classification reports and cross-validation scores
6. **Code Quality**: Improved code documentation and formatting
7. **Modularity**: Better separation of concerns and more maintainable code structure

## 📊 Supported Datasets

1. **Iris Dataset**
   - 4 features
   - 3 classes
   - 150 samples

2. **Wine Dataset**
   - 13 features
   - 3 classes
   - 178 samples

3. **Breast Cancer Dataset**
   - 30 features
   - 2 classes
   - 569 samples

## 🔍 Classifier Details

1. **K-Nearest Neighbors**
   - Non-parametric, instance-based learning
   - Suitable for small to medium-sized datasets
   - Works well with normalized features

2. **Gaussian Mixture Model**
   - Probabilistic model
   - Uses EM algorithm for fitting
   - Good for finding hidden patterns in data

3. **Support Vector Classification**
   - Maximum margin classifier
   - Effective in high-dimensional spaces
   - Versatile through kernel functions

## 📈 Performance Metrics

The application provides several metrics to evaluate classifier performance:

- Cross-validation scores
- Test accuracy
- Detailed classification report
  - Precision
  - Recall
  - F1-score
- Confusion matrix visualization
- BIC scores for model selection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Original codebase by [Original Author]
- Enhanced and modernized with Streamlit integration
- Built using scikit-learn and other open-source libraries

## 📞 Contact

For questions and feedback, please open an issue in the GitHub repository.

---

Built with ❤️ using Python and Streamlit
