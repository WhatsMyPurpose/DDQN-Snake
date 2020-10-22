# DDQL-Snake
Using Double deep Q-learning to train an agent to play snake.


<img src="https://github.com/WhatsMyPurpose/DDQL-Snake/blob/main/Snake-Videos/github-vid.gif" width="600"/>

## Snake Actions
- Left - 0
- Right - 1
- Up - 2
- Down - 3

## Returned Snake States
- If the food is somewhere above the snake's head - (0/1) <br>
- If the food is somewhere to the right of the snake's head - (0/1) <br>
- If there is an obstacle (snake body or wall) directly above the snake's head - (0/1) <br>
- If there is an obstacle directly below the snake's head - (0/1) <br>
- If there is an obstacle directly to the right of the snake's head - (0/1) <br>
- If there is an obstacle directly to the left of the snake's head - (0/1) <br>

## Snake Rewards
- Eating some food (+1)
- Dying (-10)
- Moving closer towards food (+0.1)
- Moving further away from food (-0.1)
