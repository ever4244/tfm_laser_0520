import os, sys
# hack to prevent ModuleNotFoundError with multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from . import laser_lstm
from . import laser_task
from . import cross_entropy_dist
from . import label_smoothed_cross_entropy_dist
from . import translation_lw
from . import language_pair_dataset_lw
from . import transformer_lw
from . import multilingual_translation_lw
from . import multilingual_transformer_lw
