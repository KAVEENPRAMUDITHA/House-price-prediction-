import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def test_data_loading():
    """Test if data can be loaded correctly"""
    try:
        data = pd.read_csv('data/Boston.csv')
        print(f"✅ Data loaded successfully: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_model_loading():
    """Test if model can be loaded correctly"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully: {type(model)}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_prediction(data, model):
    """Test if model can make predictions"""
    try:
        # Use first row as test data
        test_data = data.drop('MEDV', axis=1).iloc[0:1]
        prediction = model.predict(test_data)[0]
        print(f"✅ Prediction successful: ${prediction:.2f}k")
        return True
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def test_metrics_calculation(data, model):
    """Test if metrics can be calculated"""
    try:
        X = data.drop('MEDV', axis=1)
        y = data['MEDV']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Metrics calculated successfully:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return False

def main():
    print("🧪 Testing Boston Housing Price Predictor App Components")
    print("=" * 60)
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        return
    
    # Test model loading
    model = test_model_loading()
    if model is None:
        return
    
    # Test prediction
    test_prediction(data, model)
    
    # Test metrics calculation
    test_metrics_calculation(data, model)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
