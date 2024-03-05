import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

def main():
    # env = gym.make("CustomHopper-source-v0")
    env = gym.make('CustomHopper-target-v0')

    print("State space:", env.observation_space)  # state-space
    print("Action space:", env.action_space)  # action-space
    print("Dynamics parameters:", env.get_parameters())  # masses of each link of the Hopper

    """
        Dynamic parameters name: ('torso', 'thigh', 'leg', 'foot')

        Target dynamics parameters: [3.53429174 3.92699082 2.71433605 5.0893801 ]
        Source dynamics parameters: [2.53429174 3.92699082 2.71433605 5.0893801 ]
    """

    render = True
    # Target
    # model_name = "/home/borella/Desktop/project-sim2real-borella-simone/bayrn_results/bayrn_udr_ei/checkpoints/bo_point_30_.ai"
    # Source
    # model_name = "/home/borella/Desktop/project-sim2real-borella-simone/models/ppo_source/best_model.zip"
    # Best manual UDR
    model_name = "/home/borella/Desktop/project-sim2real-borella-simone/models/ppo_udr_0-95_0-95_0-95_perc/best_model.zip"
    # Best UDR
    # model_name = "/home/borella/Desktop/project-sim2real-borella-simone/models/ppo_target/best_model.zip"
    # Best GDR
    # model_name = "/home/borella/Desktop/project-sim2real-borella-simone/bayrn_results/bayrn_nei/checkpoints/bo_point_3_.ai"

    model = PPO.load(model_name)
    n_episodes = 100

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action, _state = model.predict(state)
            state, reward, done, info = env.step(action)
            time.sleep(0.01)
            env.render(mode="human")

    # mean_reward, std_reward = evaluate_policy(
    #     model=model,
    #     env=env,
    #     n_eval_episodes=n_episodes,
    #     deterministic=True,
    #     render=render,
    # )
    # print(f"{mean_reward=}")
    # print(f"{std_reward=}")


if __name__ == "__main__":
    main()
