# Update the imports section at the top of your file
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Added mean_absolute_error

# Rest of your imports remain the same
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """Create comprehensive feature set including advanced trend analysis"""
    # Base seasonal features
    df['season'] = pd.cut(df['observation_date'].dt.month, 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    df['is_winter'] = (df['season'] == 'Winter').astype(int)
    df['is_spring'] = (df['season'] == 'Spring').astype(int)
    df['is_summer'] = (df['season'] == 'Summer').astype(int)
    df['is_fall'] = (df['season'] == 'Fall').astype(int)
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['observation_date'].dt.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df['observation_date'].dt.month/12)
    
    # Advanced price features
    df['price_momentum'] = df['Price Per Dozen (wholesale)'].pct_change(3)
    df['price_acceleration'] = df['price_momentum'].diff()
    
    # Rolling statistics
    windows = [3, 6, 12]
    for window in windows:
        df[f'rolling_mean_{window}'] = df['Price Per Dozen (wholesale)'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Price Per Dozen (wholesale)'].rolling(window=window).std()
        
    # Trend strength indicators
    df['trend_strength'] = (df['rolling_mean_3'] - df['rolling_mean_12']).abs() / df['rolling_std_12']
    
    # Price lags
    for i in range(1, 4):
        df[f'price_lag_{i}'] = df['Price Per Dozen (wholesale)'].shift(i)
    
    # Volatility features
    df['volatility'] = df['Price Per Dozen (wholesale)'].rolling(window=3).std() / df['rolling_mean_3']
    
    if 'GASREGW' in df.columns:
        # Gas price features
        df['gas_price_change'] = df['GASREGW'].pct_change()
        df['gas_price_change_3m'] = df['GASREGW'].pct_change(3)
        df['gas_price_acceleration'] = df['gas_price_change'].diff()
        
        # Gas moving averages
        df['gas_ma_3'] = df['GASREGW'].rolling(window=3).mean()
        df['gas_ma_6'] = df['GASREGW'].rolling(window=6).mean()
        df['gas_ma_12'] = df['GASREGW'].rolling(window=12).mean()
        
        # Gas price trends
        df['gas_price_ratio'] = df['GASREGW'] / df['gas_ma_6']
        df['gas_volatility'] = df['GASREGW'].rolling(window=3).std() / df['gas_ma_3']
        
        # Cross-correlations
        df['price_gas_ratio'] = df['Price Per Dozen (wholesale)'] / df['GASREGW']
        df['price_gas_correlation'] = df['price_gas_ratio'].rolling(window=6).corr(df['gas_price_change'])
        
        # Leading indicators
        for i in range(1, 4):
            df[f'gas_lead_{i}'] = df['GASREGW'].shift(-i)
            df[f'gas_change_lead_{i}'] = df['gas_price_change'].shift(-i)
    
    return df


def get_feature_list(df):
    """Get list of features based on available columns"""
    base_features = [
        'time_index', 'month_sin', 'month_cos',
        'is_winter', 'is_spring', 'is_summer', 'is_fall',
        'price_momentum', 'price_acceleration',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'trend_strength', 'volatility',
        'price_lag_1', 'price_lag_2', 'price_lag_3'
    ]
    
    gas_features = [
        'GASREGW', 'gas_price_change', 'gas_price_change_3m',
        'gas_price_acceleration', 'gas_ma_3', 'gas_ma_6', 'gas_ma_12',
        'gas_price_ratio', 'gas_volatility', 'price_gas_ratio',
        'price_gas_correlation', 'gas_lead_1', 'gas_lead_2', 'gas_lead_3',
        'gas_change_lead_1', 'gas_change_lead_2', 'gas_change_lead_3'
    ]
    
    if 'GASREGW' in df.columns:
        return base_features + gas_features
    return base_features


# Set page configuration
st.set_page_config(page_title="Price Predictor with Seasonality Analysis", layout="wide")

def plot_historical_prices(df):
    """Create historical price trend plot with both egg and gas prices if available"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot egg prices
    ax1.plot(df['observation_date'], df['Price Per Dozen (wholesale)'],
             color='blue', label='Egg Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Egg Price ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # If gas prices are available, plot them on secondary y-axis
    if 'GASREGW' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df['observation_date'], df['GASREGW'],
                color='red', alpha=0.6, label='Gas Price')
        ax2.set_ylabel('Gas Price ($)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Historical Price Trends')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_seasonal_features(df):
    """Create seasonal features from date"""
    df['season'] = pd.cut(df['observation_date'].dt.month, 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    # Create seasonal indicators
    df['is_winter'] = (df['season'] == 'Winter').astype(int)
    df['is_spring'] = (df['season'] == 'Spring').astype(int)
    df['is_summer'] = (df['season'] == 'Summer').astype(int)
    df['is_fall'] = (df['season'] == 'Fall').astype(int)
    
    # Create cyclical features for month
    df['month_sin'] = np.sin(2 * np.pi * df['observation_date'].dt.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df['observation_date'].dt.month/12)
    
    return df

def load_and_preprocess_data(price_file, gas_file=None):
    """Load and preprocess both price and gas data with all required features"""
    try:
        # Read the price data
        df_price = pd.read_csv(price_file)
        df_price['observation_date'] = pd.to_datetime(df_price['observation_date'])
        df_price = df_price.sort_values('observation_date')
        
        if gas_file is not None:
            # Read the gas price data
            df_gas = pd.read_csv(gas_file)
            df_gas['observation_date'] = pd.to_datetime(df_gas['observation_date'])
            df_gas = df_gas.sort_values('observation_date')
            
            # Resample gas data to monthly frequency to match egg prices
            df_gas_monthly = df_gas.set_index('observation_date').resample('M').mean().reset_index()
            
            # Merge the datasets
            df = pd.merge_asof(df_price, df_gas_monthly, 
                             on='observation_date', 
                             direction='nearest')
        else:
            df = df_price
        
        # Create temporal features
        df['year'] = df['observation_date'].dt.year
        df['month'] = df['observation_date'].dt.month
        df['time_index'] = range(len(df))
        
        # Create seasonal features
        df['season'] = pd.cut(df['observation_date'].dt.month, 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Create cyclical month features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Create season indicators
        df['is_winter'] = (df['season'] == 'Winter').astype(int)
        df['is_spring'] = (df['season'] == 'Spring').astype(int)
        df['is_summer'] = (df['season'] == 'Summer').astype(int)
        df['is_fall'] = (df['season'] == 'Fall').astype(int)
        
        # Create lagged features for prices
        for i in range(1, 4):
            df[f'price_lag_{i}'] = df['Price Per Dozen (wholesale)'].shift(i)
        
        # Create rolling features for prices
        df['rolling_mean_3'] = df['Price Per Dozen (wholesale)'].rolling(window=3).mean()
        df['rolling_mean_6'] = df['Price Per Dozen (wholesale)'].rolling(window=6).mean()
        df['rolling_mean_12'] = df['Price Per Dozen (wholesale)'].rolling(window=12).mean()
        df['rolling_std_3'] = df['Price Per Dozen (wholesale)'].rolling(window=3).std()
        df['rolling_std_6'] = df['Price Per Dozen (wholesale)'].rolling(window=6).std()
        df['rolling_std_12'] = df['Price Per Dozen (wholesale)'].rolling(window=12).std()
        
        if 'GASREGW' in df.columns:
            # Gas price changes
            df['gas_price_change'] = df['GASREGW'].pct_change()
            df['gas_price_change_3m'] = df['GASREGW'].pct_change(3)
            df['gas_price_acceleration'] = df['gas_price_change'].diff()
            
            # Gas moving averages
            df['gas_ma_3'] = df['GASREGW'].rolling(window=3).mean()
            df['gas_ma_6'] = df['GASREGW'].rolling(window=6).mean()
            df['gas_ma_12'] = df['GASREGW'].rolling(window=12).mean()
            
            # Gas price trends
            df['gas_price_ratio'] = df['GASREGW'] / df['gas_ma_6']
            df['gas_volatility'] = df['GASREGW'].rolling(window=3).std() / df['gas_ma_3']
            
            # Leading indicators
            for i in range(1, 4):
                df[f'gas_lead_{i}'] = df['GASREGW'].shift(-i)
                df[f'gas_change_lead_{i}'] = df['gas_price_change'].shift(-i)

        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None



def apply_feature_weights(X, weights_dict):
    """Apply weights to features with proper type handling"""
    X_weighted = X.copy()
    
    for feature, weight in weights_dict.items():
        if feature in X_weighted.columns:
            # Ensure both the feature values and weight are float type
            X_weighted[feature] = X_weighted[feature].astype(float) * float(weight)
    
    return X_weighted

def prepare_data_for_modeling(df, weights_dict):
    """Prepare data for model training with fixed feature handling"""
    # Define base features that should always be present
    base_features = [
        'time_index', 'month_sin', 'month_cos',
        'is_winter', 'is_spring', 'is_summer', 'is_fall',
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
    ]
    
    # Gas-related features that should be present if gas data is available
    gas_features = [
        'GASREGW', 'gas_price_change', 'gas_price_change_3m',
        'gas_price_acceleration', 'gas_ma_3', 'gas_ma_6', 'gas_ma_12',
        'gas_price_ratio', 'gas_volatility',
        'gas_lead_1', 'gas_lead_2', 'gas_lead_3',
        'gas_change_lead_1', 'gas_change_lead_2', 'gas_change_lead_3'
    ] if 'GASREGW' in df.columns else []
    
    # Combine features based on data availability
    features = base_features + gas_features
    
    X = df[features].copy()
    y = df['Price Per Dozen (wholesale)']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Apply feature weights
    X_weighted = X_imputed.copy()
    for feature, weight in weights_dict.items():
        if feature in X_weighted.columns:
            X_weighted[feature] *= weight
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'features': features,
        'imputer': imputer
    }


def prepare_data_for_modeling(df, weights_dict):
    """Prepare data for model training with weighted features"""
    # Define base features that should always be present
    base_features = [
        'time_index', 'month_sin', 'month_cos',
        'is_winter', 'is_spring', 'is_summer', 'is_fall',
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12'
    ]
    
    # Gas-related features that should be present if gas data is available
    gas_features = [
        'GASREGW', 'gas_price_change', 'gas_price_change_3m',
        'gas_price_acceleration', 'gas_ma_3', 'gas_ma_6', 'gas_ma_12',
        'gas_price_ratio', 'gas_volatility',
        'gas_lead_1', 'gas_lead_2', 'gas_lead_3',
        'gas_change_lead_1', 'gas_change_lead_2', 'gas_change_lead_3'
    ]
    
    # Combine features based on data availability
    features = base_features + (gas_features if 'GASREGW' in df.columns else [])
    
    # Ensure all required features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    X = df[features]
    y = df['Price Per Dozen (wholesale)']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Apply feature weights
    X_weighted = X_imputed.copy()
    for feature, weight in weights_dict.items():
        if feature in X_weighted.columns:
            X_weighted[feature] *= weight
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'features': features,
        'imputer': imputer
    }

def evaluate_2024_predictions(predictions_df, df_2024):
    """Evaluate prediction accuracy for 2024 using data from main dataframe"""
    if df_2024 is None or df_2024.empty:
        return None
        
    # Convert prediction dates to datetime if they aren't already
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    
    # Filter 2024 data
    df_2024['observation_date'] = pd.to_datetime(df_2024['observation_date'])
    actual_2024 = df_2024[df_2024['observation_date'].dt.year == 2024]
    
    if actual_2024.empty:
        return None
    
    # Find overlapping dates
    overlapping_dates = predictions_df['Date'].isin(actual_2024['observation_date'])
    if not any(overlapping_dates):
        return None
        
    metrics = {}
    pred_data = predictions_df[overlapping_dates]
    actual_data = actual_2024.set_index('observation_date')['Price Per Dozen (wholesale)']
    actual_data = actual_data.reindex(pred_data['Date'])
    
    for model in ['Linear Regression', 'Random Forest', 'Neural Network', 'Ensemble']:
        pred_vals = pred_data[model]
        # Calculate metrics
        mape = np.mean(np.abs((actual_data - pred_vals) / actual_data)) * 100
        rmse = np.sqrt(mean_squared_error(actual_data, pred_vals))
        r2 = r2_score(actual_data, pred_vals)
        
        metrics[model] = {
            'MAPE': f'{mape:.2f}%',
            'RMSE': f'${rmse:.2f}',
            'R2': f'{r2:.3f}'
        }
    
    return pd.DataFrame(metrics).transpose()

def plot_seasonality_analysis(df):
    """Create seasonality analysis plots"""
    # Create season column if it doesn't exist
    if 'season' not in df.columns:
        df['season'] = pd.cut(df['observation_date'].dt.month, 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    fig = plt.figure(figsize=(15, 6))
    
    # Monthly box plot
    ax1 = plt.subplot(121)
    sns.boxplot(data=df, x=df['observation_date'].dt.month, 
                y='Price Per Dozen (wholesale)', ax=ax1)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price Per Dozen ($)')
    ax1.set_title('Monthly Price Distribution')
    
    # Seasonal trends
    ax2 = plt.subplot(122)
    seasonal_data = df.copy()  # Create a copy to avoid modifying original
    sns.boxplot(data=seasonal_data, x='season', y='Price Per Dozen (wholesale)', ax=ax2)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Price Per Dozen ($)')
    ax2.set_title('Seasonal Price Distribution')
    
    plt.tight_layout()
    return fig

def plot_correlation_analysis(df):
    """Create correlation analysis plots"""
    if 'GASREGW' not in df.columns:
        return None
        
    fig = plt.figure(figsize=(15, 5))
    
    # Time series comparison
    ax1 = plt.subplot(121)
    ax1.plot(df['observation_date'], df['Price Per Dozen (wholesale)'], 
            label='Egg Price', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(df['observation_date'], df['GASREGW'], 
            label='Gas Price', color='red', alpha=0.6)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Egg Price ($)', color='blue')
    ax2.set_ylabel('Gas Price ($)', color='red')
    plt.title('Egg Prices vs Gas Prices Over Time')
    
    # Scatter plot with regression line
    ax3 = plt.subplot(122)
    sns.regplot(data=df, x='GASREGW', y='Price Per Dozen (wholesale)', 
                ax=ax3, scatter_kws={'alpha':0.5})
    plt.title('Correlation between Egg and Gas Prices')
    
    correlation = df['Price Per Dozen (wholesale)'].corr(df['GASREGW'])
    plt.annotate(f'Correlation: {correlation:.2f}', 
                xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    return fig


def train_models(X_train, y_train):
    """Train multiple prediction models"""
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Neural Network
    nn_model = create_neural_network(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    return lr_model, rf_model, nn_model

def create_neural_network(input_shape):
    """Create and compile neural network model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def generate_future_features(df, features):
    """Generate future features with fixed array handling"""
    future_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS')
    periods = len(future_dates)
    
    # Create future data DataFrame with all required columns initialized
    future_data = pd.DataFrame(index=range(periods))
    
    # Initialize all features with zeros first
    for feature in features:
        future_data[feature] = 0.0
    
    # Basic features
    future_data['time_index'] = range(len(df), len(df) + periods)
    future_data['month_sin'] = np.sin(2 * np.pi * future_dates.month/12)
    future_data['month_cos'] = np.cos(2 * np.pi * future_dates.month/12)
    
    # Season indicators
    seasons = pd.cut(future_dates.month, 
                    bins=[0, 3, 6, 9, 12], 
                    labels=['Winter', 'Spring', 'Summer', 'Fall'])
    future_data['is_winter'] = (seasons == 'Winter').astype(int)
    future_data['is_spring'] = (seasons == 'Spring').astype(int)
    future_data['is_summer'] = (seasons == 'Summer').astype(int)
    future_data['is_fall'] = (seasons == 'Fall').astype(int)
    
    # Calculate recent trends from historical data
    historical_data = df[df['observation_date'].dt.year < 2024]
    if len(historical_data) == 0:
        historical_data = df
    
    last_prices = historical_data['Price Per Dozen (wholesale)'].iloc[-12:].values
    
    # Rolling statistics (ensure these are scalar values)
    for window in [3, 6, 12]:
        future_data[f'rolling_mean_{window}'] = float(np.mean(last_prices[-window:]))
        future_data[f'rolling_std_{window}'] = float(np.std(last_prices[-window:]))
    
    # Price lags (ensure these are scalar values)
    for i in range(1, 4):
        future_data[f'price_lag_{i}'] = float(last_prices[-i])
    
    if 'GASREGW' in df.columns:
        last_gas = historical_data['GASREGW'].iloc[-12:].values
        gas_trend = np.polyfit(range(len(last_gas)), last_gas, 1)[0]
        
        # Gas features (ensure these are scalar values)
        future_data['GASREGW'] = last_gas[-1] + gas_trend
        future_data['gas_price_change'] = float(gas_trend / last_gas[-1])
        future_data['gas_price_change_3m'] = float(np.mean(np.diff(last_gas[-3:])) / last_gas[-3])
        
        for window in [3, 6, 12]:
            future_data[f'gas_ma_{window}'] = float(np.mean(last_gas[-window:]))
            
        for i in range(1, 4):
            future_data[f'gas_lead_{i}'] = float(last_gas[-1] + gas_trend * i)
    
    # Ensure all features are present and have proper scalar values
    for feature in features:
        if feature not in future_data.columns:
            future_data[feature] = 0.0
        else:
            # Convert any remaining non-scalar values to scalar
            future_data[feature] = future_data[feature].astype(float)
    
    return future_dates, future_data[features]

def make_future_predictions(lr_model, rf_model, nn_model, scaler, imputer, future_features, future_dates):
    """Make predictions with proper array handling"""
    try:
        # Ensure future_features is a proper 2D array
        future_features_imputed = pd.DataFrame(
            imputer.transform(future_features), 
            columns=future_features.columns
        )
        future_features_scaled = scaler.transform(future_features_imputed)
        
        # Make predictions and ensure they are 1D arrays
        lr_pred = lr_model.predict(future_features_scaled).flatten()
        rf_pred = rf_model.predict(future_features_scaled).flatten()
        nn_pred = nn_model.predict(future_features_scaled).flatten()
        
        # Calculate ensemble predictions
        ensemble_pred = np.mean([lr_pred, rf_pred, nn_pred], axis=0)
        
        # Create DataFrame with predictions
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Linear Regression': lr_pred,
            'Random Forest': rf_pred,
            'Neural Network': nn_pred,
            'Ensemble': ensemble_pred
        })
        
        # Ensure all numeric columns are float type
        numeric_columns = ['Linear Regression', 'Random Forest', 'Neural Network', 'Ensemble']
        predictions[numeric_columns] = predictions[numeric_columns].astype(float)
        
        # Round predictions to 2 decimal places
        predictions[numeric_columns] = predictions[numeric_columns].round(2)
        
        return predictions
        
    except Exception as e:
        st.error(f"Error in make_future_predictions: {str(e)}")
        return None


def plot_predictions_chart(df, y_test, lr_pred, rf_pred, nn_pred, future_dates, future_preds):
    """Create an interactive prediction chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['observation_date'],
        y=df['Price Per Dozen (wholesale)'],
        name='Historical Prices',
        line=dict(color='black', width=2)
    ))
    
    test_dates = df['observation_date'].iloc[-len(y_test):]
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=lr_pred,
        name='Linear Regression',
        line=dict(color='blue', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=rf_pred,
        name='Random Forest',
        line=dict(color='green', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=nn_pred,
        name='Neural Network',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['Linear Regression'],
        name='LR Future',
        line=dict(color='blue', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['Random Forest'],
        name='RF Future',
        line=dict(color='green', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['Neural Network'],
        name='NN Future',
        line=dict(color='red', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['Ensemble'],
        name='Ensemble Future',
        line=dict(color='purple', width=3)
    ))
    
    fig.update_layout(
        title='Price Predictions',
        xaxis_title='Date',
        yaxis_title='Price Per Dozen ($)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600
    )
    
    return fig

def plot_feature_importance(rf_model, feature_names):
    """Plot feature importance without weights (fixed version)"""
    # Get raw importance scores
    importance = rf_model.feature_importances_
    
    # Create DataFrame with importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    ax.barh(importance_df['feature'], importance_df['importance'])
    
    plt.title('Feature Importance in Price Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return fig

def create_weight_sliders():
    """Create enhanced sliders for feature weights"""
    st.sidebar.header("Feature Weights")
    weights = {}
    
    # Seasonal weights
    st.sidebar.subheader("Seasonal Weights")
    weights['month_sin'] = st.sidebar.slider("Seasonal Cycle (Sin)", 0.0, 2.0, 1.0, 0.1)
    weights['month_cos'] = st.sidebar.slider("Seasonal Cycle (Cos)", 0.0, 2.0, 1.0, 0.1)
    
    # Season indicators
    weights['is_winter'] = st.sidebar.slider("Winter Impact", 0.0, 2.0, 1.0, 0.1)
    weights['is_spring'] = st.sidebar.slider("Spring Impact", 0.0, 2.0, 1.0, 0.1)
    weights['is_summer'] = st.sidebar.slider("Summer Impact", 0.0, 2.0, 1.0, 0.1)
    weights['is_fall'] = st.sidebar.slider("Fall Impact", 0.0, 2.0, 1.0, 0.1)
    
    # Historical price weights
    st.sidebar.subheader("Historical Price Weights")
    weights['price_lag_1'] = st.sidebar.slider("Previous Month Price", 0.0, 2.0, 1.0, 0.1)
    weights['rolling_mean_3'] = st.sidebar.slider("3-Month Average", 0.0, 2.0, 1.0, 0.1)
    weights['rolling_mean_6'] = st.sidebar.slider("6-Month Average", 0.0, 2.0, 1.0, 0.1)
    
    # Gas price weights
    st.sidebar.subheader("Gas Price Impact")
    weights['GASREGW'] = st.sidebar.slider("Current Gas Price", 0.0, 2.0, 1.0, 0.1)
    weights['gas_price_lag_1'] = st.sidebar.slider("1-Month Lagged Gas Price", 0.0, 2.0, 1.0, 0.1)
    weights['gas_price_lag_3'] = st.sidebar.slider("3-Month Lagged Gas Price", 0.0, 2.0, 1.0, 0.1)
    weights['gas_trend_3m'] = st.sidebar.slider("3-Month Gas Price Trend", 0.0, 2.0, 1.0, 0.1)
    
    return weights

def validate_2024_predictions(df, predictions_df):
    """
    Validate model predictions against actual 2024 data with proper array handling
    """
    # Convert all dates to datetime
    df = df.copy()
    predictions_df = predictions_df.copy()
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    
    # Filter actual 2024 data
    actual_2024 = df[df['observation_date'].dt.year == 2024].copy()
    
    if actual_2024.empty:
        st.warning("No 2024 data available for validation")
        return None, None
    
    # Merge predictions with actual data
    comparison_data = pd.merge(
        actual_2024[['observation_date', 'Price Per Dozen (wholesale)']],
        predictions_df,
        left_on='observation_date',
        right_on='Date',
        how='inner'
    )
    
    if comparison_data.empty:
        st.warning("No matching dates found between predictions and actual 2024 data")
        return None, None
    
    # Calculate validation metrics
    metrics = {}
    models = ['Linear Regression', 'Random Forest', 'Neural Network', 'Ensemble']
    
    for model in models:
        # Ensure arrays are 1D numpy arrays of float type
        actual = np.array(comparison_data['Price Per Dozen (wholesale)']).astype(float)
        predicted = np.array(comparison_data[model]).astype(float)
        
        # Calculate metrics
        try:
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            metrics[model] = {
                'MAPE (%)': round(mape, 2),
                'RMSE ($)': round(rmse, 3),
                'MAE ($)': round(mae, 3),
                'R²': round(r2, 3)
            }
        except Exception as e:
            st.error(f"Error calculating metrics for {model}: {str(e)}")
            metrics[model] = {
                'MAPE (%)': None,
                'RMSE ($)': None,
                'MAE ($)': None,
                'R²': None
            }
    
    # Create visualization
    try:
        fig = go.Figure()
        
        # Plot actual prices
        fig.add_trace(go.Scatter(
            x=comparison_data['observation_date'],
            y=comparison_data['Price Per Dozen (wholesale)'].astype(float),
            name='Actual Prices',
            line=dict(color='black', width=2)
        ))
        
        # Plot model predictions
        colors = {
            'Linear Regression': 'blue',
            'Random Forest': 'green',
            'Neural Network': 'red',
            'Ensemble': 'purple'
        }
        
        for model, color in colors.items():
            fig.add_trace(go.Scatter(
                x=comparison_data['observation_date'],
                y=comparison_data[model].astype(float),
                name=f'{model}',
                line=dict(color=color, dash='dash')
            ))
        
        fig.update_layout(
            title='2024 Predictions vs Actual Values',
            xaxis_title='Date',
            yaxis_title='Price Per Dozen ($)',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        fig = None
    
    return pd.DataFrame(metrics).transpose(), fig

def print_validation_summary(metrics):
    """Print a formatted summary of validation metrics"""
    if not metrics:
        print("No validation metrics available")
        return
    
    print("\n=== Model Validation Summary ===\n")
    
    # Find best model for each metric
    best_models = {}
    for metric in ['MAPE (%)', 'RMSE ($)', 'MAE ($)', 'R²']:
        if metric == 'R²':
            best_model = max(metrics.items(), key=lambda x: x[1][metric])[0]
        else:
            best_model = min(metrics.items(), key=lambda x: x[1][metric])[0]
        best_models[metric] = best_model
    
    # Print metrics for each model
    headers = ['Model', 'MAPE (%)', 'RMSE ($)', 'MAE ($)', 'R²']
    print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}")
    print("-" * 68)
    
    for model, model_metrics in metrics.items():
        print(
            f"{model:<20} "
            f"{model_metrics['MAPE (%)']:<12.2f} "
            f"{model_metrics['RMSE ($)']:<12.3f} "
            f"{model_metrics['MAE ($)']:<12.3f} "
            f"{model_metrics['R²']:<12.3f}"
        )
    
    print("\n=== Best Performing Models ===\n")
    for metric, model in best_models.items():
        print(f"Best {metric:<10}: {model}")
    
    # Overall recommendation
    print("\n=== Recommendation ===\n")
    mape_scores = {model: metrics[model]['MAPE (%)'] for model in metrics}
    most_accurate = min(mape_scores.items(), key=lambda x: x[1])[0]
    print(f"Based on overall performance, the {most_accurate} model shows the best accuracy")
    print(f"with a Mean Absolute Percentage Error of {mape_scores[most_accurate]:.2f}%")

# Example usage in your main application:

def main():
    """Main application function"""
    st.title("Price Predictor with Seasonality Analysis")
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        price_file = st.file_uploader("Upload Price Data (CSV)", type=['csv'])
    with col2:
        gas_file = st.file_uploader("Upload Gas Price Data (CSV)", type=['csv'])
    
    # Create weight sliders in sidebar
    weights = create_weight_sliders()
    
    if price_file is not None:
        df = load_and_preprocess_data(price_file, gas_file)
        
        if df is not None:
            st.subheader("Data Overview")
            st.write(f"Total records: {len(df)}")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Seasonality Analysis",
                "Historical Trends", 
                "Feature Importance", 
                "Price Predictions"
            ])
            
            with tab1:
                st.subheader("Seasonality Analysis")
                fig = plot_seasonality_analysis(df)
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Historical Price Trends")
                fig = plot_historical_prices(df)
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Model Analysis")
                try:
                    model_data = prepare_data_for_modeling(df, weights)
                    lr_model, rf_model, nn_model = train_models(
                        model_data['X_train_scaled'], 
                        model_data['y_train']
                    )
                    # Remove weights_applied parameter
                    fig = plot_feature_importance(
                        rf_model, 
                        model_data['features']
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error in model analysis: {str(e)}")
            
            with tab4:
                # Rest of your code remains the same...
                st.subheader("Price Predictions")
                try:
                    # Prepare data and train models
                    model_data = prepare_data_for_modeling(df, weights)
                    lr_model, rf_model, nn_model = train_models(
                        model_data['X_train_scaled'], 
                        model_data['y_train']
                    )
                    
                    # Generate predictions on test set
                    lr_predictions = lr_model.predict(model_data['X_test_scaled'])
                    rf_predictions = rf_model.predict(model_data['X_test_scaled'])
                    nn_predictions = nn_model.predict(model_data['X_test_scaled']).flatten()
                    
                    # Generate future features and predictions
                    future_dates, future_features = generate_future_features(
                        df, 
                        model_data['features']
                    )
                    
                    # Apply feature weights to future features
                    future_features = apply_feature_weights(future_features, weights)
                    
                    # Make future predictions
                    future_predictions = make_future_predictions(
                        lr_model, rf_model, nn_model,
                        model_data['scaler'], 
                        model_data['imputer'],
                        future_features, 
                        future_dates
                    )
                    
                    # Plot predictions
                    fig = plot_predictions_chart(
                        df, 
                        model_data['y_test'],
                        lr_predictions,
                        rf_predictions, 
                        nn_predictions,
                        future_dates, 
                        future_predictions
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Future Price Predictions")
                    st.dataframe(future_predictions)
                    
                    # Model validation for 2024
                    st.subheader("2024 Model Validation")
                    validation_metrics, validation_fig = validate_2024_predictions(df, future_predictions)
                    
                    if validation_metrics is not None and validation_fig is not None:
                        # Display validation visualization
                        st.plotly_chart(validation_fig, use_container_width=True)
                        
                        # Display metrics
                        st.write("### Model Performance Metrics")
                        st.dataframe(validation_metrics)
                        
                        # Find best performing model
                        best_mape = validation_metrics['MAPE (%)'].min()
                        best_model = validation_metrics['MAPE (%)'].idxmin()
                        
                        st.write(f"""
                        ### Validation Summary
                        - Best performing model: **{best_model}**
                        - Prediction accuracy (MAPE): {best_mape:.2f}%
                        - This means predictions deviate from actual values by {best_mape:.2f}% on average
                        """)
                        
                        # Additional metrics interpretation
                        st.write("""
                        ### Metrics Explanation
                        - **MAPE**: Mean Absolute Percentage Error (lower is better)
                        - **RMSE**: Root Mean Square Error in dollars (lower is better)
                        - **MAE**: Mean Absolute Error in dollars (lower is better)
                        - **R²**: R-squared value, measure of fit (closer to 1 is better)
                        """)
                    
                except Exception as e:
                    st.error(f"Error in predictions or validation: {str(e)}")

if __name__ == "__main__":
    main()