import numpy as np

# ----------------
board_shape = (6,7)
# ----------------

def find_available(col):
    pos = np.where(col==0)[0]
    if pos.size == 0:
        return -1
    else:
        return np.max(pos)

# Empty board:
board = np.zeros(board_shape)
finished = False
iter_count = 0
while not finished:
    iter_count += 1
    print(board)
    available_pos = np.apply_along_axis(func1d=find_available, axis=0, arr=board)
    print(type(available_pos))
    action = int(input("choose column:"))
    if action > 6 or action < 0:
        print("action {0} is not available".format(action))
        print("choose again...")
        continue
    else:
        if available_pos[action] == -1:
            print("...")
            print("Column already full")
            print("...")
            continue
        else:
            if iter_count % 2 == 1:
                board[available_pos[action], action] = 1
            elif iter_count % 2 == 0:
                board[available_pos[action], action] = -1
    if iter_count == 10:
        print(board)
        finished = True