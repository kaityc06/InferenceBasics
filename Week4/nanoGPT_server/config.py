"""Settings for server.py. Edit the values here and restart the server."""

# network
HOST = "127.0.0.1"
PORT = 8000

# model
MODEL_TYPE = "gpt2"  # which runner in runners.py to use
MODEL = "gpt2"   # one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl
DEVICE = None    # None = auto-pick cuda > mps > cpu; or set "cuda", "mps", "cpu"

# batching
BATCH_SIZE = 4   # N: number of prompts batched together
MAX_WAIT = None  # seconds to wait for a full batch before running a partial one; None = always wait for N

# generation (temperature and top-k apply to the whole batch)
MAX_NEW_TOKENS = 32  # tokens generated for every request (clients can't change this)
TEMPERATURE = 1.0
TOP_K = None         # None = sample from the full distribution
