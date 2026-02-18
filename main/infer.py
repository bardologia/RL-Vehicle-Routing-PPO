from tools.config import Config
from core.environment import Environment
from infer import ModelInference
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
    print(result)
