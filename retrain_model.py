import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def retrain_model():
    """Retrain the model with current scikit-learn version"""
    print("🔄 Retraining Boston Housing Price Predictor Model")
    print("=" * 60)
    
    # Load data
    try:
        data = pd.read_csv('data/dataset.csv')
        print(f"✅ Data loaded successfully: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Prepare features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    print(f"📊 Features: {X.shape[1]}")
    print(f"🎯 Target: {y.name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📈 Training samples: {X_train.shape[0]}")
    print(f"🧪 Test samples: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    
    # Save the model
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("\n💾 Model saved successfully as 'model.pkl'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    # Test loading the model
    try:
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test prediction
        test_prediction = loaded_model.predict(X_test.iloc[0:1])[0]
        print(f"✅ Model loading test successful: ${test_prediction:.2f}k")
        
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False
    
    print("\n🎉 Model retraining completed successfully!")
    return True

if __name__ == "__main__":
    retrain_model()
