# Stock Price Prediction Application

A modular application for predicting stock prices using machine learning techniques.

## Directory Structure

- `models/`: Core prediction models and algorithms
- `interfaces/`: User interfaces (CLI, GUI, etc.)
- `run.sh`: Universal launcher script for macOS/Linux
- `cli.py`: Command-line interface entry point
- `gui.py`: Web GUI interface entry point

## Getting Started

### Installation

1. Make sure you have Python 3.7+ installed
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn yfinance fastapi uvicorn matplotlib
   ```

### Running the Application

#### Command Line Interface

```bash
# Basic usage
python cli.py --ticker ALO.PA --horizon 5

# With more options
python cli.py --ticker MSFT --horizon 10 --threshold 0.65 --model ensemble --plot
```

#### Web Interface

```bash
# Start the web server
python gui.py
```
Then open your browser to http://127.0.0.1:8000

Alternatively, use the shell script (on macOS/Linux):
```bash
chmod +x run.sh
./run.sh
```