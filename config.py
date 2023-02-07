MAX_D = 5.0
HIDDEN_CHANNELS = 4
TIME_DIM = HIDDEN_CHANNELS * 2
REDUCE = "sum"
NUM_LAYERS = 2
T = 500
BATCH_SIZE = 8
LEARNING_RATE = 3e-4

config_dict = {name.lower(): value for name, value in locals().items()}
