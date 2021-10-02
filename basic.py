import argparse
from habitat_baselines.common.environments import LocomotionRLEnv
from habitat_baselines.config.default import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)
    env = LocomotionRLEnv(config=config)
    obs = env.reset(render_episode=True)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    print('Episode done!')
