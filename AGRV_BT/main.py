# Runs on Binance TESTNET (demo) when TESTNET=true. No local paper-trading simulation.
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

# auth flag: set True if we detect -2015
auth_blocked = False

# ---------- CONFIG ----------
EXCHANGE_ID = "binance"   # ccxt ID for spot; testnet via set_sandbox_mode(True)
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    # add/remove pairs as needed
]
# default symbol kept for backward compatibility in some logs
SYMBOL = SYMBOLS[0]
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

# Also log to console so smoke test feedback is visible immediately
_root = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in _root.handlers):
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _root.addHandler(_ch)

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

    if not _looks_like_binance_key(API_KEY) or not _looks_like_binance_key(API_SECRET):
        logging.warning("API key/secret format invalid (should be alphanumeric, no quotes/spaces, correct testnet keys).")

    try:
        ping = exchange.publicGetPing()
        logging.info(f"Exchange ping OK: {ping}")
    except Exception:
        logging.exception("Public ping failed")

    try:
        info = exchange.publicGetExchangeInfo()
        logging.info(f"ExchangeInfo OK (symbols: {len(info.get('symbols', []))})")
    except Exception:
        logging.exception("publicGetExchangeInfo failed")

    try:
        b = exchange.fetch_balance()
        usdt = (b.get('total') or {}).get('USDT', 0)
        where = "TESTNET" if TESTNET else "PROD"
        logging.info(f"USDT balance ({where}): {usdt}")
    except Exception as e:
        msg = str(e)
        if '"code":-2014' in msg or 'API-key format invalid' in msg:
            logging.error("Auth error -2014: API-key format invalid. Remove quotes/spaces, ensure correct TESTNET keys, and re-copy from portal.")
        elif '"code":-2015' in msg or 'Invalid API-key, IP, or permissions' in msg:
            logging.error("Auth error -2015: Invalid API-key/IP/permissions. Check IP whitelist, enable Read/Trade, and confirm testnet keys.")
            global auth_blocked
            auth_blocked = True
        else:
            logging.exception("Balance check failed")

# Helper: fetch balance USD-equivalent
def get_usd_balance():
    global auth_blocked
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
            logging.exception("fetch_balance failed; returning 0 for safety")
        return 0.0

# Ensure order amount respects Binance filters (minNotional, minQty, precision)
def clamp_amount_to_filters(symbol: str, price: float, amount: float) -> float:
    try:
        # Make sure markets are loaded so limits/precision exist
        if not getattr(exchange, 'markets', None):
            exchange.load_markets()
        market = exchange.market(symbol)
        # Min notional (cost)
        min_cost = None
        try:
            min_cost = (market.get('limits') or {}).get('cost', {}).get('min', None)
        except Exception:
            min_cost = None
        if min_cost is not None and price is not None:
            if amount * price < float(min_cost):
                amount = (float(min_cost) / price) * 1.001  # small safety buffer
        # Min quantity (amount)
        min_qty = None
        try:
            min_qty = (market.get('limits') or {}).get('amount', {}).get('min', None)
        except Exception:
            min_qty = None
        if min_qty is not None:
            if amount < float(min_qty):
                amount = float(min_qty)
        # Apply exchange precision rounding using ccxt helper
        amount = float(exchange.amount_to_precision(symbol, amount))
        return max(0.0, amount)
    except Exception:
        logging.exception("Failed to clamp amount to filters; returning original amount")
        return amount

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
    if auth_blocked:
        logging.warning("Auth blocked (-2015). Fix API permissions/IP whitelist (and ensure TESTNET keys).")
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
    global last_order_time
    try:
        logging.info(f"Placing market order {side} {amount} {symbol}")
        order = exchange.create_order(symbol, type='market', side=side, amount=amount)
        last_order_time = datetime.utcnow()
        return order
    except Exception:
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
                logging.info(f"Stop loss hit for {sym}. Closing long...")
                place_market_order(sym, 'sell', pos['amount'])
                pnl = (last_price - pos['entry']) * pos['amount']
                cumulative_lost_since_start += pnl
                logging.info(f"{sym} STOP closed PnL={pnl:.6f}")
                del open_positions[sym]
        elif pos['side'] == 'sell':
            if last_price >= pos['entry'] * (1 + STOP_LOSS_PCT):
                logging.info(f"Stop loss hit for short {sym}. Closing short...")
                place_market_order(sym, 'buy', pos['amount'])
                pnl = (pos['entry'] - last_price) * pos['amount']
                cumulative_lost_since_start += pnl
                logging.info(f"{sym} SHORT STOP closed PnL={pnl:.6f}")
                del open_positions[sym]

# Preflight checks to verify connectivity and market readiness without placing orders
def preflight_checks():
    try:
        exchange.load_markets()
        for sym in SYMBOLS:
            assert sym in exchange.markets, f"Symbol {sym} not in exchange markets"
            ticker = exchange.fetch_ticker(sym)
            _ = ticker.get('last')
            bars = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=10)
            assert isinstance(bars, list) and len(bars) > 0, f"No OHLCV data for {sym}"
        bal = get_usd_balance()
        logging.info(f"[PREFLIGHT] OK: markets loaded, ticker & OHLCV available for {len(SYMBOLS)} symbols, USDT balance={bal}")
    except Exception:
        logging.exception("[PREFLIGHT] Failed. Check TESTNET, API keys, or symbols list.")
        raise

# Main loop
def main_loop():
    global cumulative_lost_since_start
    mode = "TESTNET (Binance demo)" if TESTNET else "PROD (REAL MONEY!)"
    logging.info(f"Starting aggressive bot on {mode}.")
    logging.info(f"Toggles -> TESTNET={TESTNET}")
    try:
        logging.info(f"File: {__file__} | CWD: {os.getcwd()}")
    except Exception:
        pass
    try:
        logging.info(f"Sandbox active? {getattr(exchange, 'sandbox', None)}")
        logging.info(f"Base URLs: {getattr(exchange, 'urls', None)}")
    except Exception:
        pass
    check_connection()
    preflight_checks()
    while True:
        try:
            if not can_trade():
                time.sleep(1)
                continue
            usd_bal = get_usd_balance()  # fetch once per cycle
            for sym in SYMBOLS:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    price = float(ticker['last'])
                    signal = simple_momentum_signal(sym, TIMEFRAME)
                    if signal == "buy":
                        size = compute_order_size(usd_bal, price)
                        size = clamp_amount_to_filters(sym, price, size)
                        if size > 0 and (size * price) >= MIN_ORDER_USD:
                            order = place_market_order(sym, 'buy', size)
                            if order:
                                open_positions[sym] = {'entry': price, 'amount': size, 'side': 'buy', 'ts': datetime.utcnow()}
                                logging.info(f"Opened LONG {sym} amt={size} @ {price}")
                        else:
                            logging.info(f"{sym}: Buy skipped (size too small after filters/min notional)")
                    elif signal == "sell":
                        if sym in open_positions and open_positions[sym]['side'] == 'buy':
                            amt = open_positions[sym]['amount']
                            place_market_order(sym, 'sell', amt)
                            pnl = (price - open_positions[sym]['entry']) * amt
                            cumulative_lost_since_start += pnl
                            logging.info(f"Closed LONG {sym} amt={amt} @ {price} PnL={pnl:.6f}")
                            del open_positions[sym]
                except Exception:
                    logging.exception(f"Loop error for symbol {sym}")
            # manage stops for all symbols
            track_and_manage_positions()

            time.sleep(0.5)  # aggressive but allow rate limit
        except Exception as e:
            logging.exception("Main loop error")
            time.sleep(1)

if __name__ == "__main__":
    main_loop()