# aggressive_bot.py
import os, time, math, logging
from dotenv import load_dotenv
import ccxt
from datetime import datetime, timedelta

load_dotenv()  # place tes clés dans .env (ou passe autrement)

# ---------- CONFIG ----------
EXCHANGE_ID = "binance"   # exemple ; utilise testnet/paper si possible
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
RISK_PCT = 0.02           # % capital par trade (2%)
MAX_DAILY_LOSS_PCT = 0.05 # stop global si perte 5% cap
COOLDOWN_SECONDS = 2      # anti-spam
MAX_POSITION_USD = 2000   # cap fixe absolu
MIN_ORDER_USD = 10
STOP_LOSS_PCT = 0.01      # 1% stop
LOG_FILE = "aggr_bot.log"
# ----------------------------

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# create exchange (example: Binance)
exchange = ccxt.binance({
    'apiKey': os.getenv('API_KEY'),
    'secret': os.getenv('API_SECRET'),
    # pour testnet (binance) : uncomment lines below and configure
    # 'urls': {'api': 'https://testnet.binance.vision/api'},
    'enableRateLimit': True
})

# Helper: fetch balance USD-equivalent
def get_usd_balance():
    bal = exchange.fetch_balance()
    # adapte suivant l'exchange: on suppose USDT stable
    usdt = bal.get('total', {}).get('USDT', 0)
    return float(usdt)

# Risk manager: compute size in quote currency (USDT)
def compute_order_size(usd_balance, price):
    risk_cap = usd_balance * RISK_PCT
    size = risk_cap / price
    # enforce absolute cap:
    if (size * price) > MAX_POSITION_USD:
        size = MAX_POSITION_USD / price
    if (size * price) < MIN_ORDER_USD:
        return 0
    # round per exchange precision (very approximate)
    precision = 6
    return math.floor(size * 10**precision) / 10**precision

# Simple aggressive strategy: if price rises X% in last N candles -> buy market
def simple_momentum_signal(symbol, timeframe):
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=10)
    closes = [b[4] for b in bars]
    if len(closes) < 5:
        return None
    # momentum: last close vs average of prior 5
    last = closes[-1]
    avg_prev = sum(closes[-6:-1]) / 5
    pct = (last - avg_prev) / avg_prev
    # thresholds aggressive: small moves trigger
    if pct > 0.002:   # 0.2% move -> buy
        return "buy"
    if pct < -0.002:  # -0.2% -> sell (close long)
        return "sell"
    return None

# Order executor with risk checks
last_order_time = datetime.min
daily_start = datetime.utcnow().date()
cumulative_pnl = 0.0
cumulative_lost_since_start = 0.0

def can_trade():
    global last_order_time, daily_start, cumulative_lost_since_start
    # reset daily counters at UTC midnight
    if datetime.utcnow().date() != daily_start:
        daily_start = datetime.utcnow().date()
        cumulative_lost_since_start = 0.0
    if (datetime.utcnow() - last_order_time).total_seconds() < COOLDOWN_SECONDS:
        return False
    usd_bal = get_usd_balance()
    # global max loss
    if cumulative_lost_since_start <= -abs(usd_bal * MAX_DAILY_LOSS_PCT):
        logging.warning("Daily loss limit hit. No more trading.")
        return False
    return True

def place_market_order(symbol, side, amount):
    global last_order_time
    try:
        logging.info(f"Placing market order {side} {amount} {symbol}")
        # real order:
        order = exchange.create_order(symbol, type='market', side=side, amount=amount)
        last_order_time = datetime.utcnow()
        return order
    except Exception as e:
        logging.exception("Order failed")
        return None

# Very simple stop loss implementation (market stop implemented via checking price feed)
open_positions = {}  # symbol -> dict with entry price, amount, side, timestamp

def track_and_manage_positions():
    global cumulative_lost_since_start
    for sym, pos in list(open_positions.items()):
        ticker = exchange.fetch_ticker(sym)
        last_price = float(ticker['last'])
        if pos['side'] == 'buy':
            if last_price <= pos['entry'] * (1 - STOP_LOSS_PCT):
                logging.info(f"Stop loss hit for {sym}. Closing...")
                # compute size approximate
                place_market_order(sym, 'sell', pos['amount'])
                pnl = (last_price - pos['entry']) * pos['amount']
                cumulative_lost_since_start += pnl
                del open_positions[sym]
        elif pos['side'] == 'sell':
            if last_price >= pos['entry'] * (1 + STOP_LOSS_PCT):
                logging.info(f"Stop loss hit for short {sym}. Closing...")
                place_market_order(sym, 'buy', pos['amount'])
                pnl = (pos['entry'] - last_price) * pos['amount']
                cumulative_lost_since_start += pnl
                del open_positions[sym]

# Main loop
def main_loop():
    global cumulative_lost_since_start
    logging.info("Starting aggressive bot in PAPER mode. DO NOT USE WITH REAL MONEY UNTIL TESTED.")
    while True:
        try:
            if not can_trade():
                time.sleep(1)
                continue
            usd_bal = get_usd_balance()
            ticker = exchange.fetch_ticker(SYMBOL)
            price = float(ticker['last'])
            signal = simple_momentum_signal(SYMBOL, TIMEFRAME)
            if signal == "buy":
                size = compute_order_size(usd_bal, price)
                if size > 0:
                    order = place_market_order(SYMBOL, 'buy', size)
                    if order:
                        open_positions[SYMBOL] = {'entry': price, 'amount': size, 'side': 'buy', 'ts': datetime.utcnow()}
            elif signal == "sell":
                # if have open long, close; else short (dangerous) — here we close longs only
                if SYMBOL in open_positions and open_positions[SYMBOL]['side'] == 'buy':
                    amt = open_positions[SYMBOL]['amount']
                    place_market_order(SYMBOL, 'sell', amt)
                    pnl = (price - open_positions[SYMBOL]['entry']) * amt
                    cumulative_lost_since_start += pnl
                    del open_positions[SYMBOL]
            # manage stops
            track_and_manage_positions()

            time.sleep(0.5)  # aggressive but allow rate limit
        except Exception as e:
            logging.exception("Main loop error")
            time.sleep(1)

if __name__ == "__main__":
    main_loop()