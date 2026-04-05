import deepgram
# Check available modules
print("Version:", deepgram.__version__)
print("\nTop-level exports:")
print([x for x in dir(deepgram) if not x.startswith('_')])

# Check listen module
from deepgram import DeepgramClient
c = DeepgramClient.__new__(DeepgramClient)
print("\nClient methods:")
print([x for x in dir(c) if not x.startswith('_')])

# Check if listen has async websocket
if hasattr(c, 'listen'):
    print("\nListen attrs:", [x for x in dir(c.listen) if not x.startswith('_')])
