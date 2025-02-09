from random import randint

GLOBAL_SEED = randint(0, 2**64-1)
N_QUBITS = 2

#NONANOM_PATH = "data/ligo/raw/nonanom_5000.csv"
#ANOM_PATH = "data/ligo/raw/anom.csv"

DATA_PATH_BASE = "data/ligo/raw/"

NONANOM_PATH = "data/ligo/raw/noise_unfiltered.pkl"
ANOM_PATH = "data/ligo/raw/events_unfiltered.pkl"

#NONANOM_PATH = "data/ligo/raw/noise.pkl"
#ANOM_PATH = "data/ligo/raw/events.pkl"

NONANOM_PATH_SYNTHETIC =  "data/synthetic/nonanom.pickle"
ANOM_PATH_SYNTHETIC = "data/synthetic/anom.pickle"

TRAIN_PATH = "data/train.pickle"
VALIDATE_PATH = "data/validate.pickle"
NONANOM_TEST_PATH = "data/nonanom_test.pickle"
ANOM_TEST_PATH = "data/anom_test.pickle"

TRAINING_RESULTS_PATH = "training_results.pickle"
TUNING_RESULTS_PATH = "tuning_results.pickle"
TESTING_RESULTS_PATH = "testing_results.pickle"

NONANOM_SCORES_PATH = "nonanom_scores.pickle"
ANOM_SCORES_PATH = "anom_scores.pickle"
