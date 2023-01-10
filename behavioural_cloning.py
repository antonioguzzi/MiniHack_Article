import gym
from gym.wrappers import FlattenObservation
import minihack
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

from stable_baselines3 import PPO, A2C, SAC, TD3

ENV_ID = "MiniHack-Room-5x5-v0"


# ---------------- THIS CLASS DESCRIBE THE EXPERT DATASET ---------------- #
class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]

    def __len__(self):
        return len(self.observations)


# ------------------------------------------------------------------------ #


def pretrain_agent(
        student,
        env,
        batch_size=64,
        epochs=1000,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        test_batch_size=64,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    # Train the model
    def train(model, device, train_loader, optimizer):
        model.train()  # Change model mode
        loss = 0
        # print(len(train_loader))
        for batch_idx, (exp_obs, exp_act) in enumerate(train_loader):
            exp_obs, exp_act = exp_obs.to(device), exp_act.to(device)
            optimizer.zero_grad()

            # Predict action based on observation
            if isinstance(env.action_space, gym.spaces.Box):
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(exp_obs)
                else:
                    action = model(exp_obs)
                action_prediction = action.double()
            else:
                dist = model.get_distribution(exp_obs)
                action_prediction = dist.distribution.logits
                exp_act = exp_act.long()

            # Computing distance between expert action and  model prediction
            loss = criterion(action_prediction, exp_act)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(exp_obs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(), ))
        loss /= len(train_loader)
        print(f"[TRAIN SET] Average loss: {loss}")

    # Test the model
    def test(model, device, test_loader):
        model.eval()  # Change model mode
        test_loss = 0

        with th.no_grad():
            for exp_obs, exp_act in test_loader:
                exp_obs, exp_act = exp_obs.to(device), exp_act.to(device)

                # Predict action based on observation
                if isinstance(env.action_space, gym.spaces.Box):
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(exp_obs)
                    else:
                        action = model(exp_obs)
                    action_prediction = action.double()
                else:
                    dist = model.get_distribution(exp_obs)
                    action_prediction = dist.distribution.logits
                    exp_act = exp_act.long()

                # Computing distance between expert action and  model prediction
                test_loss = criterion(action_prediction, exp_act)
        test_loss /= len(test_loader.dataset)
        print(f"[TEST SET] Average loss: {test_loss}")

    # Use PyTorch `DataLoader` to load previously created `ExpertDataset` for training and testing
    train_loader = th.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True, **kwargs)

    # Define an Optimizer and a learning rate scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train the policy model
    for epoch in range(1, epochs + 1):
        print(f"-------------------------------")
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
    print(f"-------------------------------")
    # Implant the trained policy back into the RL student agent
    student.policy = model


def evaluate(model, env, agent_name, steps=100, epochs=5):
    print(f"Evaluating {agent_name} on env [{ENV_ID}]...")

    for e in range(epochs):
        obs = env.reset()
        eval_rewards = []
        print(f"----------- EPOCH {e + 1} -----------")
        # If agent doesn't reach the goal until steps=100 the cycle will end
        for i in range(steps):
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            eval_rewards.append(reward)
            if done and i >= 99:
                print(f"{agent_name} doesn't reach the goal.")
                break
            elif done and i < 99:
                print(f"Goal reached by {agent_name} in {i} steps.")
                break
        print(f"Average reward per step: {sum(eval_rewards) / len(eval_rewards)}.")
    print(f"-------------------------------")


def create_expert_dataset():
    expert_data = './dataset/expert_dataset/expert_data.npz'
    expert_observations = np.load(expert_data)['expert_observations']
    expert_actions = np.load(expert_data)['expert_actions']

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)

    # 80% train 20% test 12
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(expert_dataset, [train_size, test_size])

    return train_expert_dataset, test_expert_dataset


def build_env():
    environment = gym.make(ENV_ID, observation_keys=("chars", "colors"), )
    environment = FlattenObservation(environment)
    print(f"Env [{ENV_ID}] info:\n"
          f"\tObservation space: {environment.observation_space.shape, type(environment.observation_space)}\n"
          f"\tAction space shape:{environment.action_space.shape, type(environment.action_space)}\n")
    return environment


if __name__ == "__main__":
    # Environment Initialization
    env = build_env()

    # Build Train & Test set
    train_set, test_set = create_expert_dataset()
    print(f"Train set dimension: {len(train_set)}")
    print(f"Test set dimension: {len(test_set)}\n")

    # Create the student & Test knowledge before training
    print(f"Creating the student...")
    a2c_student = A2C('MlpPolicy', env, verbose=1)
    print(f"\n[EVALUATING THE STUDENT BEFORE TRAINING]")
    evaluate(a2c_student, env, 'STUDENT')

    print(f"\n[TRAINING THE STUDENT]")
    pretrain_agent(
        a2c_student,
        env,
        epochs=5,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=5,
        no_cuda=True,
        seed=1,
        batch_size=32,
        test_batch_size=32,
    )

    # Evaluate the student on env
    print(f"\n[EVALUATING THE STUDENT AFTER TRAINING]")
    evaluate(a2c_student, env, 'STUDENT')
    a2c_student.save("a2c_student")
    env.close()
