from Agents.HumanUser import User
from Agents.AlphaZero import Agent
from Agents._memory import Memory
from Agents._self_play_funcs import playMatches
from Games.Connect4 import Game
from Connect4_MCTreeSearch_config import Config

from Connect4_MCTreeSearch_model import Connect4_MCTreeSearch_model

human_VS_human = True
human_VS_agent = False
agent_VS_agent = False

# -----------
episodes = 1
goes_first = 0
# -----------
env = Game()
config = Config()
memory = Memory(config)

if human_VS_human:
    player1 = User('player 1', env.state_size, env.action_size)
    player2 = User('player 2', env.state_size, env.action_size)

elif human_VS_agent:
    player1 = User('player 1', env.state_size, env.action_size)
    player2 = Agent('player 2', env.state_size, env.action_size, 
                    model = Connect4_MCTreeSearch_model, 
                    config = config)

elif agent_VS_agent:
    player2 = Agent('player 1', env.state_size, env.action_size, 
                    model = Connect4_MCTreeSearch_model, 
                    config = config)
    player2 = Agent('player 2', env.state_size, env.action_size, 
                    model = Connect4_MCTreeSearch_model, 
                    config = config)


scores, memory, points =  playMatches(player1, 
                                      player2, 
                                      episodes = episodes, 
                                      goes_first = goes_first, 
                                      env = env,
                                      memory = memory)
print(scores)
print(points)