import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Load datasets
iris = datasets.load_iris()
wine = datasets.load_wine()
cancer = datasets.load_breast_cancer()

# Create a dictionary of datasets
datasets_dict = {'Iris': iris, 'Wine': wine, 'Breast Cancer': cancer}

# Create a dictionary of classifiers
classifiers_dict = {'K-Nearest Neighbours': KNeighborsClassifier(), 
                    'Gaussian Mixture Model': GaussianMixture(), 
                    'Support Vector Classification': SVC()}


def preprocess_data(dataset):
    # Separate features and target
    X = dataset.data
    y = dataset.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def classify():
    # Get selected dataset and classifier
    dataset = datasets_dict[var_dataset.get()]
    classifier_name = var_classifier.get()

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
    
    if classifier_name == 'Gaussian Mixture Model':
        # Use GMM to generate new features
        gmm = GaussianMixture(n_components=len(np.unique(y_train)))
        gmm.fit(X_train)
        X_train = gmm.predict_proba(X_train)
        X_test = gmm.predict_proba(X_test)

        # Use another classifier for actual classification
        classifier = KNeighborsClassifier()
    else:
        classifier = classifiers_dict[classifier_name]

    # Perform cross validation and fit classifier
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    classifier.fit(X_train, y_train)

    # Calculate accuracy and confusion matrix
    accuracy = classifier.score(X_test, y_test)
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    

    # Display results
    print('Cross Validation Scores:', scores)
    print('Accuracy:', accuracy)
    print('Confusion Matrix:\n', cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4,4))  # Adjust figure size here
    sns.heatmap(cm, annot=True, ax=ax)
    plt.xlabel('Predicted label')
    plt.ylabel('Truth label')
    plt.show()


def plot_graph():
    # Get selected dataset
    dataset_name = var_dataset.get()
    dataset = datasets_dict[dataset_name]

    # Preprocess data
    X, y = preprocess_data(dataset)

    # Initialize lists to store BIC scores and number of components
    bic_scores = []
    n_components = list(range(1, 7))

    # Calculate BIC score for each number of components
    for n in n_components:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    # Create a new figure and plot data
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(n_components, bic_scores, color='blue')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('BIC score')
    
    # Set title of the graph
    ax.set_title(f'BIC score per model for {dataset_name} dataset')

    # Create a new window
    new_window = Toplevel(root)
    
    # Create a canvas and add it to the GUI
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Create GUI window
root = Tk()

# Set window size
root.geometry("400x300")  # Width x Height

# Set window title
root.title("Classification App")

# Create a style object
style = ttk.Style(root)

# Set a theme
style.theme_use("clam")

# Configure style
style.configure('TButton', font=('Arial', 10), foreground='black')
style.configure('TLabel', font=('Arial', 10), foreground='blue')

# Create variables for selected dataset and classifier
var_dataset = StringVar(root)
var_classifier = StringVar(root)

# Set default values
var_dataset.set('Iris')
var_classifier.set('K-Nearest Neighbours')

# Create dropdown menus for dataset and classifier selection
dataset_menu = ttk.Combobox(root, textvariable=var_dataset, values=list(datasets_dict.keys()))
classifier_menu = ttk.Combobox(root, textvariable=var_classifier, values=list(classifiers_dict.keys()))

# Create button to run classification and plot graph 
classify_button = ttk.Button(root, text="Run Classification", command=classify)
plot_button = ttk.Button(root, text="Plot Graph", command=plot_graph)


# Place widgets in window using grid layout 
ttk.Label(root, text="Select Dataset:").grid(row=0,column=0, padx=10, pady=10)
dataset_menu.grid(row=0,column=1, padx=10, pady=10)
ttk.Label(root, text="Select Classifier:").grid(row=1,column=0, padx=10, pady=10)
classifier_menu.grid(row=1,column=1, padx=10, pady=10)
classify_button.grid(row=2,column=0, padx=10, pady=10)
plot_button.grid(row=2,column=1, padx=10, pady=10)

# Start GUI loop
root.mainloop()
