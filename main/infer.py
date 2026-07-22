import os
import sys

proj_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from tools.config import Config
from tools.logger import Logger
from core.environment import Environment
from core.inference import ModelInference
from core.model import Policy

def main(config):
    env = Environment(config)
    env.reset()
    initial_state = env.current_state.copy()

    model = Policy(config)
    model.load("model.pt", "./checkpoints")
    inference = ModelInference(model, env, max_steps=50)
    result = inference.run(initial_state)
    return result

if __name__ == "__main__":
    config = Config()
    result = main(config)

    logger = Logger(name="inference")
    logger.kv_table(result.summary(), title="Inference Summary")
