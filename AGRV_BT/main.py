# aggressive_bot.py
# Supports TESTNET and PAPER_TRADING (set PAPER_TRADING=true in .env to simulate without private API).
import os, time, math, logging, re
from dotenv import load_dotenv, find_dotenv
import ccxt
from datetime import datetime, timedelta

load_dotenv(find_dotenv(), override=True)  # load .env from project root reliably

# --- Read & sanitize API keys (trim hidden spaces/newlines) ---
RAW_API_KEY = os.getenv('API_KEY') or ''
RAW_API_SECRET = os.getenv('API_SECRET') or ''
API_KEY = RAW_API_KEY.strip()
API_SECRET = RAW_API_SECRET.strip()

# Hard guard: stop early if keys are clearly invalid/missing
if not API_KEY or len(API_KEY) < 16:
    logging.critical("API_KEY missing or too short. Ensure your .env is found and API_KEY is set (testnet key).")
    raise SystemExit(1)
if not API_SECRET or len(API_SECRET) < 16:
    logging.critical("API_SECRET missing or too short. Ensure your .env is found and API_SECRET is set (testnet secret).")
    raise SystemExit(1)

# Validate format (Binance keys are alphanum; length typically 32–128 chars)
def _looks_like_binance_key(s: str) -> bool:
    # Public API keys are typically long and alphanumeric (Binance often uses 64+ chars).
    return bool(re.fullmatch(r'[A-Za-z0-9]{30,128}', s or ''))

if not _looks_like_binance_key(API_KEY):
    logging.warning(f"API_KEY format suspicious (len={len(API_KEY)}). Check for spaces/quotes or wrong environment (testnet vs prod).")
if len(API_SECRET) < 30:
    logging.warning(f"API_SECRET length looks short (len={len(API_SECRET)}). Re-copy secret carefully (can contain symbols).")

# --- Runtime toggles from .env ---
TESTNET = (os.getenv('TESTNET', 'true').lower() == 'true')
PAPER_TRADING = (os.getenv('PAPER_TRADING', 'false').lower() == 'true')
try:
    PAPER_USDT_START = float(os.getenv('PAPER_USDT', '10000'))
except Exception:
    PAPER_USDT_START = 10000.0

# local state for paper mode
paper_balance_usdt = PAPER_USDT_START
auth_blocked = False  # set True if we detect -2015

# Simulate orders only: use real public endpoints and (optionally) real balance,
# but do NOT send private create_order() calls. Useful to test order logic
# while still using the live market data.
SIMULATE_ORDERS_ONLY = (os.getenv('SIMULATE_ORDERS_ONLY', 'false').lower() == 'true')

# store simulated orders when SIMULATE_ORDERS_ONLY is enabled
simulated_orders = []

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

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})
# Let ccxt switch all endpoints to Binance testnet automatically (if TESTNET)
if TESTNET:
    exchange.set_sandbox_mode(True)
# Allow a wider recvWindow to avoid timestamp drift auth errors on testnet
exchange.options['recvWindow'] = 50000
# Sync timestamp with Binance to avoid signature/401 errors
exchange.options['adjustForTimeDifference'] = True
try:
    exchange.load_time_difference()
except Exception as e:
    logging.exception("load_time_difference failed (continuing)")

# Prevent ccxt from calling SAPI currencies endpoint (not supported on testnet)
exchange.options['fetchCurrencies'] = False
exchange.options['warnOnFetchCurrencies'] = False

def check_connection():
    ak = os.getenv('API_KEY') or ''
    logging.info(f"API key loaded? len={len(ak)}, suffix={ak[-4:] if ak else 'None'}")
    sk = API_SECRET or ''
    logging.info(f"API secret loaded? len={len(sk)}, suffix={sk[-4:] if sk else 'None'}")
    if PAPER_TRADING:
        logging.info(f"PAPER_TRADING enabled. Using simulated balance: {PAPER_USDT_START} USDT. Skipping private auth checks.")
        try:
            ping = exchange.publicGetPing()
            logging.info(f"Testnet/public ping OK (paper): {ping}")
        except Exception as e:
            logging.exception("Public ping failed even in paper mode")
        try:
            info = exchange.publicGetExchangeInfo()
            logging.info(f"ExchangeInfo OK (paper): {len(info.get('symbols', []))} symbols")
        except Exception as e:
            logging.exception("publicGetExchangeInfo failed in paper mode")
        return
    if not _looks_like_binance_key(API_KEY) or not _looks_like_binance_key(API_SECRET):
        logging.warning("API key/secret format invalid (should be alphanumeric, no quotes/spaces, correct testnet keys).")
    try:
        ping = exchange.publicGetPing()
        logging.info(f"Testnet ping OK: {ping}")
    except Exception as e:
        logging.exception("Public ping failed on testnet")

    try:
        info = exchange.publicGetExchangeInfo()
        logging.info(f"ExchangeInfo OK (symbols: {len(info.get('symbols', []))})")
    except Exception as e:
        logging.exception("publicGetExchangeInfo failed on testnet")

    try:
        b = exchange.fetch_balance()
        usdt = (b.get('total') or {}).get('USDT', 0)
        logging.info(f"USDT balance (testnet): {usdt}")
    except Exception as e:
        msg = str(e)
        if '"code":-2014' in msg or 'API-key format invalid' in msg:
            logging.error("Auth error -2014: API-key format invalid. Remove quotes/spaces, ensure correct TESTNET keys, and re-copy from portal.")
        elif '"code":-2015' in msg or 'Invalid API-key, IP, or permissions' in msg:
            logging.error("Auth error -2015: Invalid API-key/IP/permissions. Check IP whitelist, enable Read/Trade, and confirm testnet keys.")
            global auth_blocked
            auth_blocked = True
        else:
            logging.exception("Balance check failed on testnet")

# Helper: fetch balance USD-equivalent
def get_usd_balance():
    global paper_balance_usdt, auth_blocked
    if PAPER_TRADING:
        return float(paper_balance_usdt)
    try:
        bal = exchange.fetch_balance()
        usdt = (bal.get('total') or {}).get('USDT', 0) or (bal.get('free') or {}).get('USDT', 0) or 0
        return float(usdt)
    except Exception as e:
        msg = str(e)
        if '"code":-2014' in msg or 'API-key format invalid' in msg:
            logging.error("fetch_balance auth -2014: API-key format invalid (likely stray spaces/quotes or wrong keys).")
        elif '"code":-2015' in msg or 'Invalid API-key, IP, or permissions' in msg:
            logging.error("fetch_balance auth -2015: Invalid API-key/IP/permissions (check whitelist & permissions).")
            auth_blocked = True
        else:
            logging.exception("fetch_balance failed on testnet; returning 0 for safety")
        return 0.0

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
    global last_order_time, daily_start, cumulative_lost_since_start, auth_blocked
    # reset daily counters at UTC midnight
    if datetime.utcnow().date() != daily_start:
        daily_start = datetime.utcnow().date()
        cumulative_lost_since_start = 0.0
    if (datetime.utcnow() - last_order_time).total_seconds() < COOLDOWN_SECONDS:
        return False
    if auth_blocked and not PAPER_TRADING:
        logging.warning("Auth blocked (-2015). Enable PAPER_TRADING=true in .env or fix API permissions/IP whitelist.")
        return False
    usd_bal = get_usd_balance()
    if usd_bal <= 0:
        logging.warning("USDT balance is 0 on testnet (auth or faucet issue). Skipping trading until resolved.")
        return False
    # global max loss
    if cumulative_lost_since_start <= -abs(usd_bal * MAX_DAILY_LOSS_PCT):
        logging.warning("Daily loss limit hit. No more trading.")
        return False
    return True

def place_market_order(symbol, side, amount):
    # If either full PAPER_TRADING or simulate-orders-only is enabled, simulate order locally.
    if PAPER_TRADING or SIMULATE_ORDERS_ONLY:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
        except Exception:
            price = None
        notional = amount * (price if price else 0.0)
        side_str = f"[SIMULATED] {side.upper()} {amount} {symbol} @ {price if price else 'NA'} (notional≈{notional:.2f} USDT)"
        logging.info(side_str)

        # If full paper trading, update the paper balance (real simulation).
        global paper_balance_usdt, simulated_orders
        if PAPER_TRADING:
            if price:
                if side == 'buy':
                    paper_balance_usdt = max(0.0, paper_balance_usdt - notional)
                elif side == 'sell':
                    paper_balance_usdt = paper_balance_usdt + notional
            order = {'id': 'paper-order', 'side': side, 'symbol': symbol, 'amount': amount, 'price': price, 'notional': notional, 'simulated': True}
            simulated_orders.append(order)
            return order

        # If SIMULATE_ORDERS_ONLY (but not PAPER_TRADING), record simulated order but do NOT alter paper balance.
        order = {'id': 'sim-only-order', 'side': side, 'symbol': symbol, 'amount': amount, 'price': price, 'notional': notional, 'simulated': True}
        simulated_orders.append(order)
        return order

    # Otherwise perform a real order via the exchange
    global last_order_time
    try:
        logging.info(f"Placing market order {side} {amount} {symbol}")
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
    logging.info(f"Toggles -> TESTNET={TESTNET}, PAPER_TRADING={PAPER_TRADING}, PAPER_USDT_START={PAPER_USDT_START}")
    check_connection()
    while True:
        try:
            if not can_trade():
                time.sleep(1)
                continue
            usd_bal = get_usd_balance()
            if PAPER_TRADING and int(time.time()) % 5 == 0:
                logging.info(f"[PAPER] Simulated USDT balance: {usd_bal}")
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