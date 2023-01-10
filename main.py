import random
from ttyrec import TTYRecReader
from tqdm import tqdm
import glob
import os
import action_catcher as ac
import numpy as np


# With this method we parse every single frame in the .ttyrec recording,
# so we extract the (obs,action) couple visited/performed by the player
def build_trajectories(recordings_dir):
    pre_p = tuple()
    first_it = True
    last_it = True
    observations = list()
    actions = list()

    messages = set()
    messages_writer = open('messages.txt', 'w')

    # ------------------------ FOR EACH FILE ----------------------- #
    for filename in glob.iglob(recordings_dir, recursive=True):
        if os.path.isdir(filename) or not filename.endswith('.ttyrec.bz2'):
            continue

        reader = TTYRecReader(filename)
        # --------------------- FOR EACH FRAME --------------------- #
        for frame in tqdm(reader):
            message: str = frame.message.strip()

            if not frame.done:
                # Guess player actions during the game
                if first_it:
                    observation = frame.generate_observation()
                    observations.append(observation)
                    pre_p = frame.agent_position
                    first_it = False
                else:
                    movement = ac.guess_movements(pre_p, frame.agent_position)
                    if movement is not None:  # The player has performed a movement
                        observation = frame.generate_observation()
                        observations.append(observation)
                        actions.append(movement)
                    pre_p = frame.agent_position

            elif frame.done and last_it:
                last_it = False
                actions.append(random.randint(0, 7))

            if message not in messages:  # Duplicate messages not allowed
                messages_writer.write(f'{message}\n')
                messages_writer.flush()
            messages.add(message)
        # ---------------------------------------------------------- #
        first_it = True
        last_it = True
    # ---------------------------------------------------------------- #
    return observations, actions


def build_expert_dataset(loc, obs: list, act: list):
    arr_observations = np.array(obs)
    arr_actions = np.array(act)
    # print(arr_actions, type(arr_actions), arr_actions.shape)
    # print(arr_observations, type(arr_observations), arr_observations.shape)
    np.savez_compressed(loc, expert_actions=arr_actions, expert_observations=arr_observations, )


def main():
    recordings_dir = './dataset/**'
    expert_dataset = './dataset/expert_data'

    observations, actions = build_trajectories(recordings_dir)
    # print(observations)
    # print(actions)
    build_expert_dataset(expert_dataset, observations, actions)


if __name__ == "__main__":
    main()
