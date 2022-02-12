# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 18 Aug 2021 03:16:52 PM CEST
# Description: Conditional helper functions, such as time-dependant
# functions.
# ======================================================================
import time

def generate_timer(delay = 60):
    """
    Generate a timer that returns true, while active and false
    as when the defined time has been met.
    
    Args:
        delay: delay in seconds
    """
    now = time.time()       # Get time in UNIX Epoch time
    end_time = now + delay  # Set endtime

    def timer():
        return time.time() < end_time

    return timer

