# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 18 Aug 2021 03:21:56 PM CEST
# Description: Logger class for convenient logging.
# ======================================================================
import time
class Logger:
    def __init__(self):
        self.creation_date = time.time()

    def log(self, text: str):
        print(text)

