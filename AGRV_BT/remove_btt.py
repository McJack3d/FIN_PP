from dotenv import load_dotenv
load_dotenv()

from binance_trade_bot.database import Database
from binance_trade_bot.logger import Logger
from binance_trade_bot.config import Config
from binance_trade_bot.models.coin import Coin  # <-- Import Coin model

logger = Logger()
config = Config()
db = Database(logger, config)

with db.db_session() as session:
    btt_coin = session.query(Coin).filter_by(symbol="BTT").first()
    if btt_coin:
        session.delete(btt_coin)
        session.commit()
        print("Removed BTT from coins table.")
    else:
        print("BTT not found in coins table.")