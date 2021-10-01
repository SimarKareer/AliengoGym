from habitat_baselines.common.environments import LocomotionRLEnv
from habitat import Config

def main():
    config = Config({"TASK_CONFIG": {"num_joints": 12}})
    env = LocomotionRLEnv(config)
    obs = env.reset(render_episode=True)

    done = False
    while not done:
        action = env.action_space.sample()
        # actionDict = {"action": action}
        obs, reward, done, info = env.step(action=action)
    
    print("Episode Done!")

main()