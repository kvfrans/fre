# Unsupervised Zero-Shot RL via Functional Reward Representations
Code for "Unsupervised Zero-Shot RL via Functional Reward Representations"


### Abstract
Can we pre-train a generalist agent from a large amount of unlabeled offline trajectories such that it can be immediately adapted to any new downstream tasks in a zero-shot manner?
In this work, we present a \emph{functional} reward encoding (FRE) as a general, scalable solution to this *zero-shot RL* problem.
Our main idea is to learn functional representations of any arbitrary tasks by encoding their state-reward samples using a transformer-based variational auto-encoder.
This functional encoding not only enables the pre-training of an agent from a wide diversity of general unsupervised reward functions, but also provides a way to solve any new downstream tasks in a zero-shot manner, given a small number of reward-annotated samples.
We empirically show that FRE agents trained on diverse random unsupervised reward functions can generalize to solve novel tasks in a range of simulated robotic benchmarks, often outperforming previous zero-shot RL and offline RL methods.

### Code Instructions
First install the dependencies in the `deps` folder.
```
cd deps
conda env create -f environment.yml
```

For the ExORL experiments, you will need to first download the data using [these instructions](https://github.com/denisyarats/exorl).
Then, download the [auxilliary offline data] and place it in the `data/` folder. 

To run the code for the experiments, use the following commands.

```
# AntMaze
python experiment/run_pre.py --env_name antmaze-large-diverse-v2
# ExORL
python experiment/run_pre.py --env_name dmc_walker_walk --agent.warmup_steps 1000000 --max_steps 2000000
python experiment/run_pre.py --env_name dmc_cheetah_run --agent.warmup_steps 1000000 --max_steps 2000000
# Kitchen
python experiment/run_pre.py --env_name kitchen-mixed-v0 --agent.warmup_steps 1000000 --max_steps 2000000
```
