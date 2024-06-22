import pyrallis

from src.configs.train_config import TrainConfig
from src.training.trainer import ConTEXTure


@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = ConTEXTure(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.paint()


if __name__ == '__main__':
    main()
