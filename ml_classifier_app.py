import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any
import pickle
import os

class MLClassifier:
    def __init__(self):
        # Load datasets
        self.datasets_dict = {
            'Iris': datasets.load_iris(),
            'Wine': datasets.load_wine(),
            'Breast Cancer': datasets.load_breast_cancer()
        }
        
        # Initialize classifiers
        self.classifiers_dict = {
            'K-Nearest Neighbours': KNeighborsClassifier(),
            'Gaussian Mixture Model': GaussianMixture(),
            'Support Vector Classification': SVC(probability=True)
        }
        
        self.scaler = StandardScaler()
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data by scaling features.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and target
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          classifier_name: str) -> Dict[str, Any]:
        """
        Train the selected classifier and evaluate its performance.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            classifier_name (str): Name of the classifier to use
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if classifier_name == 'Gaussian Mixture Model':
            # Use GMM for feature transformation
            gmm = GaussianMixture(n_components=len(np.unique(y_train)))
            gmm.fit(X_train)
            X_train_prob = gmm.predict_proba(X_train)
            X_test_prob = gmm.predict_proba(X_test)
            
            # Use KNN for classification
            classifier = KNeighborsClassifier()
            X_train, X_test = X_train_prob, X_test_prob
        else:
            classifier = self.classifiers_dict[classifier_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
        
        # Train classifier
        classifier.fit(X_train, y_train)
        
        # Get predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = classifier.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'cv_scores': cv_scores,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def calculate_bic_scores(self, X: np.ndarray) -> Tuple[list, list]:
        """
        Calculate BIC scores for different numbers of components.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            Tuple[list, list]: Lists of component numbers and corresponding BIC scores
        """
        n_components = list(range(1, 7))
        bic_scores = []
        
        for n in n_components:
            gmm = GaussianMixture(n_components=n)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
        
        return n_components, bic_scores

def main():
    st.set_page_config(
        page_title="ML Classifier App",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Machine Learning Classifier Application")
    st.markdown("""
    This application demonstrates various classification algorithms on classic datasets.
    Choose a dataset and a classifier to see the results!
    """)
    
    # Initialize classifier
    classifier = MLClassifier()
    
    # Sidebar for user input
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        options=list(classifier.datasets_dict.keys())
    )
    
    # Classifier selection
    clf_name = st.sidebar.selectbox(
        "Select Classifier",
        options=list(classifier.classifiers_dict.keys())
    )
    
    # Get selected dataset
    dataset = classifier.datasets_dict[dataset_name]
    
    # Display dataset information
    st.header(f"Dataset: {dataset_name}")
    st.write(f"Number of features: {dataset.data.shape[1]}")
    st.write(f"Number of samples: {dataset.data.shape[0]}")
    st.write(f"Number of classes: {len(np.unique(dataset.target))}")
    
    # Preprocess data
    X, y = classifier.preprocess_data(dataset.data, dataset.target)
    
    # Train and evaluate
    if st.sidebar.button("Run Classification"):
        with st.spinner("Training and evaluating..."):
            results = classifier.train_and_evaluate(X, y, clf_name)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                st.write(f"Cross-validation scores: {results['cv_scores'].mean():.3f} (±{results['cv_scores'].std():.3f})")
                st.write(f"Test accuracy: {results['accuracy']:.3f}")
                st.text("Classification Report:")
                st.text(results['classification_report'])
            
            with col2:
                st.subheader("Confusion Matrix")
                # Create confusion matrix heatmap using plotly
                fig = go.Figure(data=go.Heatmap(
                    z=results['confusion_matrix'],
                    x=[f"Class {i}" for i in range(len(np.unique(y)))],
                    y=[f"Class {i}" for i in range(len(np.unique(y)))],
                    text=results['confusion_matrix'],
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    hoverongaps=False,
                    colorscale='Blues'
                ))
                
                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    xaxis={'side': 'bottom'}
                )
                
                st.plotly_chart(fig)
    
    # Plot BIC scores
    if st.sidebar.button("Plot BIC Scores"):
        with st.spinner("Calculating BIC scores..."):
            n_components, bic_scores = classifier.calculate_bic_scores(X)
            
            fig = go.Figure(data=[
                go.Bar(x=n_components, y=bic_scores)
            ])
            
            fig.update_layout(
                title=f"BIC Scores for {dataset_name} Dataset",
                xaxis_title="Number of Components",
                yaxis_title="BIC Score"
            )
            
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()