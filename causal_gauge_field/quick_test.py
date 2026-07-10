import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from causal_gauge_field.utils.config import load_config
from causal_gauge_field.utils.logger import setup_logger
from causal_gauge_field.npnw.story_generator import StoryGenerator
from causal_gauge_field.npnw.tokenizer import NPNWTokenizer
from causal_gauge_field.models.transformer import CausalTransformer
from causal_gauge_field.experiments.trainer import Trainer, StoryDataset
from torch.utils.data import DataLoader

config = load_config()
config["data"]["num_stories"] = 200
config["training"]["max_epochs"] = 5
config["training"]["patience"] = 3

torch.manual_seed(42)
np.random.seed(42)

logger = setup_logger("QuickTest")
logger.info("Quick test: data gen + training")

gen = StoryGenerator(config, seed=42)
(train_pos, train_neg), (val_pos, val_neg), (test_pos, test_neg) = gen.generate_dataset(200, 3)
logger.info(f"train_pos={len(train_pos)}, train_neg={len(train_neg)}, test_pos={len(test_pos)}")

tok = NPNWTokenizer()
config["model"]["vocab_size"] = tok.vocab_size

all_train = train_pos + train_neg
all_val = val_pos + val_neg
train_ds = StoryDataset(all_train, tok, config["model"]["max_seq_len"])
val_ds = StoryDataset(all_val, tok, config["model"]["max_seq_len"])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = CausalTransformer(config)
logger.info(f"Model params: {model.count_parameters()}")

trainer = Trainer(config, model)
history = trainer.train_full(train_loader, val_loader, lambda_value=0.0)
logger.info(f"Training done. Final train_loss={history['train_loss'][-1]:.4f}")
print("QUICK TEST PASSED")