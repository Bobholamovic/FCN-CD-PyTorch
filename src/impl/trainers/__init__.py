from core.misc import R
from .cd_trainer import CDTrainer

__all__ = []

trainer_switcher = R['Trainer_switcher']
# Append the (pred, trainer) pairs to trainer_switcher
trainer_switcher.add_item(lambda C: not C['tb_on'] or C['dataset'] != 'OSCD', CDTrainer)