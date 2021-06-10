# rl_sampling
Redoing of my Reinforcement learning adaptive sampling master thesis project in torch from scratch

# Data preparation
We need reference renders in order to train adaptive sampling and denoising algorithms

- Planning:
| scene name/spp |  8 x 4  |  16 x 16  |  32 x 16  |  64 x 16  |  128 x 16  |  256 x 8  |  512 x 4  |  1024 x 2  |  2024  |  32768  |  65536  |
|  bathroom      |         |           |           |           |            |           |           |            |        |         |         |
|  living_room   |         |           |           |           |            |           |           |            |        |         |         |
|                |         |           |           |           |            |           |           |            |        |         |         |
|                |         |           |           |           |            |           |           |            |        |         |         |
|                |         |           |           |           |            |           |           |            |        |         |         |