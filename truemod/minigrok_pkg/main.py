import nest_asyncio
try: nest_asyncio.apply()
except: pass
from minigrok.app import _launch
if __name__ == '__main__': _launch()
