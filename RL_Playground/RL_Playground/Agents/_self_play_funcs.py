import random
# playMatches

def playMatches(player1, player2, episodes, goes_first, env, memory = None, render = False):
    """
    Returns:
    --------
    scores : dict
        number of wins for players and number of draws.

    memory : object
        bla

    points : 
        results for each match as a list.
    """
    scores = {player1.name : 0, 'drawn' : 0, player2.name : 0}
    points = {player1.name : [], player2.name : []}

    for episode in range(episodes):

        state = env.reset()

        done = 0
        turn = 0

        if goes_first == 0:
            firstPlayer = random.randint(0,1)*2 - 1
        else:
            firstPlayer = goes_first

        if firstPlayer == 1:
            players = {1 : player1,
                      -1 : player2}
        else:
            players = {1 : player2,
                      -1 : player1}

        if render:
            env.gameState.render()

        while not done:
            turn = turn + 1
            
            action, pi = players[state.playerTurn].act(state)

            if memory is not None:
                memory.commit_shortMemory(state, pi)

            state, value, done, _ = env.step(action) # value here is immediate reward

            if render:
                env.gameState.render()
            
        if done == 1:
            if memory is not None:
                memory.add_value_shortMemory(value, state.playerTurn) # (state, looser)
                memory.short_to_longMemory() # clear_shortMemory is caled inside

            if value == 1: # First wins
                scores[players[state.playerTurn].name] += 1

            elif value == -1: # Second wins
                scores[players[-state.playerTurn].name] += 1

            else: # Drawn
                scores['drawn'] += 1

            pts = state.score
            points[players[state.playerTurn].name].append(pts[0])
            points[players[-state.playerTurn].name].append(pts[1])

    return (scores, memory, points)



if __name__ == '__main__':
    from Agents.HumanUser import User
    from Agents._memory import Memory
    from Games.Connect4 import Game
    from Connect4_MCTreeSearch_config import Config

    env = Game()
    player1 = User('player 1', env.state_size, env.action_size)
    player2 = User('player 2', env.state_size, env.action_size)

    config = Config()
    memory = Memory(config)

    scores, memory, points =  playMatches(player1, 
                                          player2, 
                                          episodes = 1, 
                                          goes_first = 0, 
                                          env = env,
                                          memory = memory)

    print(scores)
    print(memory.longMemory)
    print(points)