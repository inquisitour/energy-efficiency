import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Dict, Any
import logging
from datetime import datetime

class EnhancedDataPreprocessor:
    """Enhanced data preprocessing class with robust error handling and validation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        self.scaler = None
        self.preprocessor = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and constraints
        
        Args:
            df: Input DataFrame
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Check for missing values
            missing_stats = df.isnull().sum()
            if missing_stats.any():
                self.logger.warning(f"Missing values detected:\n{missing_stats[missing_stats > 0]}")
            
            # Validate value ranges
            if df['energy_kWh'].min() < 0:
                raise ValueError("Negative energy values detected")
                
            if df['temperature_C'].min() < -50 or df['temperature_C'].max() > 50:
                raise ValueError("Temperature values outside reasonable range (-50°C to 50°C)")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate entries")
            
            # Validate timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_gaps = df['timestamp'].diff().dt.total_seconds() / 3600
            irregular_intervals = time_gaps[time_gaps != 1].count()
            if irregular_intervals > 0:
                self.logger.warning(f"Found {irregular_intervals} irregular time intervals")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
            
    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data quality metrics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing quality metrics
        """
        metrics = {
            'completeness': 1 - df.isnull().mean().mean(),
            'value_ranges': {
                'energy_kwh_range': [df['energy_kWh'].min(), df['energy_kWh'].max()],
                'temperature_range': [df['temperature_C'].min(), df['temperature_C'].max()]
            },
            'unique_ratio': df.nunique() / len(df),
            'timestamp_continuity': 1 - (df['timestamp'].diff().dt.total_seconds() / 3600 != 1).mean()
        }
        
        return metrics

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        # Extract basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        # Add rolling statistics
        df['energy_ma_24h'] = df['energy_kWh'].rolling(window=24, min_periods=1).mean()
        df['energy_std_24h'] = df['energy_kWh'].rolling(window=24, min_periods=1).std()
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with sophisticated strategies
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Forward fill short gaps
        df = df.fillna(method='ffill', limit=3)
        
        # For remaining gaps, use similar-day-of-week average
        for column in ['energy_kWh', 'temperature_C']:
            if df[column].isnull().any():
                df[column] = df[column].fillna(
                    df.groupby([df['timestamp'].dt.dayofweek, 
                              df['timestamp'].dt.hour])[column].transform('mean')
                )
        
        # Any remaining missing values filled with column median
        df = df.fillna(df.median())
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit preprocessor and transform data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed DataFrame, preprocessing metrics)
        """
        try:
            # Validate data
            self.validate_data(df)
            
            # Calculate initial quality metrics
            initial_metrics = self.calculate_quality_metrics(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Add temporal features
            if self.config.get('TEMPORAL_FEATURES', True):
                df = self.add_temporal_features(df)
            
            # Setup column transformer
            numeric_features = ['energy_kWh', 'temperature_C'] + \
                             [col for col in df.columns if col.startswith(('hour_', 'day_'))]
            
            categorical_features = [col for col in df.columns if df[col].dtype == 'category']
            
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', 'passthrough', categorical_features)
            ])
            
            # Fit and transform
            transformed_data = self.preprocessor.fit_transform(df)
            
            # Calculate final quality metrics
            final_metrics = self.calculate_quality_metrics(df)
            
            # Save preprocessor
            joblib.dump(self.preprocessor, self.config['PREPROCESSOR_PATH'])
            
            # Prepare feature names
            feature_names = numeric_features + categorical_features
            
            # Create output DataFrame
            transformed_df = pd.DataFrame(
                transformed_data,
                columns=feature_names,
                index=df.index
            )
            
            metrics = {
                'initial_quality': initial_metrics,
                'final_quality': final_metrics,
                'n_features': len(feature_names),
                'feature_names': feature_names
            }
            
            return transformed_df, metrics
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if self.preprocessor is None:
            try:
                self.preprocessor = joblib.load(self.config['PREPROCESSOR_PATH'])
            except:
                raise ValueError("Preprocessor not fitted and no saved preprocessor found")
                
        transformed_data = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()
        
        return pd.DataFrame(
            transformed_data,
            columns=feature_names,
            index=df.index
        )