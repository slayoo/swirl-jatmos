import os
import sys
# Analogous to what is said in if __name__ == '__main__' blocks
if 'JAX_ENABLE_X64' not in os.environ:
    os.environ['JAX_ENABLE_X64'] = str(True)

# Workarounding option parsing clash between pytest and driver.py
sys.argv = [sys.argv[0]]