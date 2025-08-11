# 🏠 Boston Housing Price Predictor

A comprehensive machine learning application for predicting housing prices in Boston using the Boston Housing dataset. This project demonstrates a complete ML pipeline from data exploration to model deployment using Streamlit.

## 📋 Project Overview

This application provides an interactive web interface for exploring the Boston Housing dataset and making real-time predictions of housing prices. The project includes:

- **Data Exploration**: Comprehensive analysis of the Boston Housing dataset
- **Interactive Visualizations**: Dynamic charts and plots using Plotly
- **Model Prediction**: Real-time price predictions with user input
- **Model Performance**: Detailed evaluation metrics and comparisons
- **Batch Prediction**: Support for bulk predictions via CSV upload

## 🎯 Key Features

### 📊 Data Exploration
- Dataset overview with shape, columns, and data types
- Interactive data filtering options
- Missing values analysis
- Statistical summaries

### 📈 Interactive Visualizations
- Distribution plots for all features
- Correlation matrix heatmap
- Interactive scatter plots
- Feature analysis with box plots

### 🔮 Model Prediction
- Real-time price predictions
- Input validation and error handling
- Confidence scores
- Batch prediction support

### 📋 Model Performance
- Comprehensive evaluation metrics
- Actual vs predicted visualizations
- Feature importance analysis
- Model comparison charts

## 🏗️ Project Structure

```
boston-housing-predictor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained machine learning model
├── data/
│   └── Boston.csv        # Boston Housing dataset
├── notebooks/
│   └── model_training.ipynb  # Jupyter notebook for model training
└── README.md             # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd boston-housing-predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

## 📊 Dataset Information

The Boston Housing dataset contains 506 samples with 14 features:

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)² where Bk is the proportion of blacks by town |
| LSTAT | % lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000s (Target) |

## 🎮 Usage Guide

### Navigation
The application features a sidebar navigation with five main sections:

1. **🏠 Home**: Project overview and quick start guide
2. **📊 Data Exploration**: Dataset analysis and filtering
3. **📈 Visualizations**: Interactive charts and plots
4. **🔮 Model Prediction**: Real-time price predictions
5. **📋 Model Performance**: Model evaluation and metrics

### Making Predictions

1. Navigate to the "🔮 Model Prediction" section
2. Enter housing characteristics using the input widgets
3. Click "🚀 Predict Price" to get instant predictions
4. View confidence scores and feature impact analysis

### Batch Predictions

1. Prepare a CSV file with the required features
2. Upload the file in the batch prediction section
3. Click "Predict for all samples"
4. Download the results with predictions

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning algorithms
- **joblib**: Model persistence

### Model Information
- **Algorithm**: Random Forest Regressor
- **Training Samples**: 506
- **Features**: 13
- **Target Variable**: MEDV (Price in $1000s)
- **Performance**: R² Score ~0.89

## 🌐 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Configure deployment settings
   - Deploy!

### Deployment Requirements
- All dependencies listed in `requirements.txt`
- Model file (`model.pkl`) included in repository
- Dataset file (`data/Boston.csv`) included in repository

## 📈 Model Performance

The trained model achieves the following performance metrics:

- **R² Score**: 0.89
- **Mean Squared Error**: 8.45
- **Mean Absolute Error**: 2.34
- **Root Mean Squared Error**: 2.91

## 🔧 Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Model Retraining
1. Update the Jupyter notebook in `notebooks/`
2. Retrain the model
3. Save the new model as `model.pkl`
4. Update performance metrics in the app

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Boston Housing dataset from UCI Machine Learning Repository
- Streamlit for the amazing web framework
- Plotly for interactive visualizations
- Scikit-learn for machine learning algorithms

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Contact the development team

---

**Built with ❤️ using Streamlit | Machine Learning Model Deployment Project** 