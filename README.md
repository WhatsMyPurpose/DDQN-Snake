# DDQL-Snake
Using Double deep Q-learning to train an agent to play snake.


<img src="https://github.com/WhatsMyPurpose/DDQL-Snake/blob/main/Snake-Videos/github-vid.gif" width="600"/>

## Snake Actions
- Left - 0
- Right - 1
- Up - 2
- Down - 3

## Snake States
- If the food is somewhere above the snakes head - (0/1) <br>
- If the food is somewhere to the right of snakes head - (0/1) <br>
- If there is an obstacle (snake body or wall) directly above the snakes head - (0/1) <br>
- If there is an obstacle directly below the snakes head - (0/1) <br>
- If there is an obstacle directly to the right of the snakes head - (0/1) <br>
- If there is an obstacle directly to the left of the snakes head - (0/1) <br>
