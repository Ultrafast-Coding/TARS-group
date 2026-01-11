import time
from contextlib import contextmanager

@contextmanager
def timer(description: str = "timer name"):
    """
    debug use, shows time taken for given codeblock
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{description} processed in {end - start:.4f} seconds.")

# example:
# with timer('timer name'):
#     target_function()