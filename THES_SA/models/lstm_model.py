"""
LSTM Forecasting Models (Phase 3)
Implements Baseline LSTM (price-only) and Sentiment-Augmented LSTM.
Uses Keras/TensorFlow with 80/20 chronological train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import pickle
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    Trains and evaluates Baseline vs Sentiment-Augmented LSTM models.
    Produces predictions for both 1-day and 5-day forward returns.
    """

    # Feature groups
    BASELINE_FEATURES = [
        'Close', 'Volume', 'Daily_Return', 'Log_Return',
        'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Realized_Volatility'
    ]

    SENTIMENT_FEATURES = [
        'Sentiment_Index', 'Sentiment_Momentum'
    ]

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.lstm_config = self.config['lstm']
        self.model_config = self.config['model']

        self.sequence_length = self.lstm_config['sequence_length']
        self.hidden_size = self.lstm_config['hidden_size']
        self.num_layers = self.lstm_config['num_layers']
        self.dropout = self.lstm_config['dropout']
        self.epochs = self.lstm_config['epochs']
        self.batch_size = self.lstm_config['batch_size']
        self.learning_rate = self.lstm_config['learning_rate']
        self.patience = self.lstm_config['early_stopping_patience']
        self.val_split = self.lstm_config['validation_split']
        self.test_size = self.model_config['test_size']

        self.processed_path = Path(self.config['paths']['processed'])
        self.models_path = Path(self.config['paths']['models'])
        self.results_path = Path(self.config['paths']['results'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.core_set = self.config['tickers']['core_set']
        self.benchmark_set = self.config['tickers']['benchmark_set']

        logger.info(f"LSTMForecaster initialized (seq_len={self.sequence_length}, "
                    f"hidden={self.hidden_size}, layers={self.num_layers})")

    def _build_model(self, n_features: int, model_name: str = "lstm"):
        """Build a Keras LSTM model"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

        model = Sequential(name=model_name)
        model.add(Input(shape=(self.sequence_length, n_features)))

        # Stack LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            model.add(LSTM(
                self.hidden_size,
                return_sequences=return_sequences,
                name=f'lstm_{i}'
            ))
            model.add(Dropout(self.dropout, name=f'dropout_{i}'))

        # Output layer: single value (predicted return)
        model.add(Dense(1, name='output'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM input.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)

        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological 80/20 split"""
        df = df.sort_values('Date')
        split_idx = int(len(df) * (1 - self.test_size))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def train_single_model(self,
                           df_ticker: pd.DataFrame,
                           feature_cols: List[str],
                           target_col: str,
                           model_name: str) -> dict:
        """
        Train a single LSTM model for one ticker and one target.

        Args:
            df_ticker: DataFrame for a single ticker (sorted by date)
            feature_cols: List of input feature columns
            target_col: Target column name
            model_name: Name for the model

        Returns:
            Dict with model, predictions, actuals, and metadata
        """
        import tensorflow as tf

        # Check which features actually exist
        available_features = [c for c in feature_cols if c in df_ticker.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features for {model_name}: {missing}")

        if not available_features:
            logger.error(f"No features available for {model_name}")
            return {}

        # Drop rows with NaN in features or target
        df_clean = df_ticker.dropna(subset=available_features + [target_col])
        if len(df_clean) < self.sequence_length + 20:
            logger.warning(f"Not enough data for {model_name}: {len(df_clean)} rows")
            return {}

        # Split
        train_df, test_df = self._train_test_split(df_clean)

        X_train = train_df[available_features].values
        y_train = train_df[target_col].values
        X_test = test_df[available_features].values
        y_test = test_df[target_col].values

        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            logger.warning(f"Insufficient sequence data for {model_name}")
            return {}

        # Build and train
        model = self._build_model(len(available_features), model_name)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
        ]

        logger.info(f"Training {model_name}: {X_train_seq.shape[0]} train, "
                    f"{X_test_seq.shape[0]} test sequences")

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.val_split,
            callbacks=callbacks,
            verbose=0
        )

        # Predict
        y_pred = model.predict(X_test_seq, verbose=0).flatten()

        # Get corresponding dates for test predictions
        test_dates = test_df['Date'].iloc[self.sequence_length:].values

        return {
            'model': model,
            'history': history.history,
            'y_pred': y_pred,
            'y_actual': y_test_seq,
            'test_dates': test_dates,
            'features_used': available_features,
            'n_train': len(X_train_seq),
            'n_test': len(X_test_seq),
            'epochs_trained': len(history.history['loss']),
        }

    def run_experiment(self,
                       data_file: str = 'merged_dataset.csv',
                       target_col: str = 'Forward_Return_1d') -> Dict:
        """
        Run the full modeling experiment for all tickers.
        Trains Baseline and Sentiment-Augmented models for each ticker.

        Args:
            data_file: Input merged dataset filename
            target_col: Target column to predict

        Returns:
            Dict of results keyed by ticker
        """
        logger.info("=" * 70)
        logger.info(f"PHASE 3: LSTM MODELING EXPERIMENT (target: {target_col})")
        logger.info("=" * 70)

        # Load data
        data_path = self.processed_path / data_file
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            logger.info("Run Phase 2 (sentiment features) first.")
            return {}

        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])

        all_results = {}

        for ticker in df['Ticker'].unique():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training models for {ticker}")
            logger.info(f"{'='*50}")

            df_ticker = df[df['Ticker'] == ticker].sort_values('Date').copy()

            if len(df_ticker) < self.sequence_length + 50:
                logger.warning(f"Skipping {ticker}: only {len(df_ticker)} rows")
                continue

            # Baseline LSTM (price features only)
            baseline_result = self.train_single_model(
                df_ticker,
                self.BASELINE_FEATURES,
                target_col,
                f"baseline_{ticker}_{target_col}"
            )

            # Sentiment-Augmented LSTM
            augmented_features = self.BASELINE_FEATURES + self.SENTIMENT_FEATURES
            augmented_result = self.train_single_model(
                df_ticker,
                augmented_features,
                target_col,
                f"augmented_{ticker}_{target_col}"
            )

            group = 'core_set' if ticker in self.core_set else 'benchmark_set'

            all_results[ticker] = {
                'group': group,
                'baseline': baseline_result,
                'augmented': augmented_result,
            }

            if baseline_result and augmented_result:
                logger.info(f"{ticker} training complete "
                           f"(baseline epochs: {baseline_result['epochs_trained']}, "
                           f"augmented epochs: {augmented_result['epochs_trained']})")

        # Save results summary
        self._save_predictions(all_results, target_col)

        logger.info("\n" + "=" * 70)
        logger.info("LSTM MODELING COMPLETE")
        logger.info("=" * 70)

        return all_results

    def _save_predictions(self, results: Dict, target_col: str):
        """Save all predictions to CSV for analysis"""
        rows = []
        for ticker, data in results.items():
            for model_type in ['baseline', 'augmented']:
                result = data.get(model_type, {})
                if not result:
                    continue

                for i in range(len(result['y_pred'])):
                    rows.append({
                        'Ticker': ticker,
                        'Group': data['group'],
                        'Model': model_type,
                        'Target': target_col,
                        'Date': result['test_dates'][i] if i < len(result['test_dates']) else None,
                        'y_actual': result['y_actual'][i],
                        'y_pred': result['y_pred'][i],
                    })

        if rows:
            df_pred = pd.DataFrame(rows)
            output_path = self.results_path / f'predictions_{target_col}.csv'
            df_pred.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

    def save_models(self, results: Dict, target_col: str):
        """Save trained Keras models to disk"""
        for ticker, data in results.items():
            for model_type in ['baseline', 'augmented']:
                result = data.get(model_type, {})
                if not result or 'model' not in result:
                    continue

                model_path = self.models_path / f"{model_type}_{ticker}_{target_col}.keras"
                result['model'].save(model_path)
                logger.info(f"Saved model: {model_path}")

    def run_full_experiment(self) -> Dict:
        """
        Run experiments for both 1-day and 5-day horizons.

        Returns:
            Dict with results for both horizons
        """
        all_results = {}

        for target in self.model_config['target_columns']:
            results = self.run_experiment(target_col=target)
            all_results[target] = results

            if results:
                self.save_models(results, target)

        return all_results


def main():
    forecaster = LSTMForecaster()
    forecaster.run_full_experiment()


if __name__ == "__main__":
    main()
