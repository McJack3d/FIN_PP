from dotenv import load_dotenv
load_dotenv()

import os
print("API_KEY:", os.environ.get("API_KEY"))
print("API_SECRET:", os.environ.get("API_SECRET"))

from .crypto_trading import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass