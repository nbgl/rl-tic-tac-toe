# rl-tic-tac-toe

This is an idea for a project that teaches basic reinforcement learning to high schoolers. At one stage I was planning it for an [NCSS](https://ncss.edu.au/summer-school) Masterclass, but it may be too complicated. I&rsquo;m putting it on GitHub in case someone finds it useful &#x1F937;

It is aimed at bright students, who know basic Python and high-school level maths. It is dependent on base Python 3 only and does not require third-party packages. Students should be able to complete this in a day from scratch, or a few hours if they are given a template that implements the boring bits.

Students will learn the basic concepts of reinforcement learning (agent, state, action, reward) as well as basic _Q_-learning.

Of course, students vary in their capabilities. We need to cater to students who are beginners to computer science, as well as to more experienced programmers (without anyone getting lost or bored). There are multiple stages to the exercise. A student who completes at least two can say that they used reinforcement learning to play tic-tac-toe!

Those stages are described below. Sample solutions are included in the repo. I had fun implementing the solutions in ways I thought were neat. I expect students to not know Python&rsquo;s more advanced functionality.

# Stages

## Stage 1: interface
The students make a framework for playing tic-tac-toe. They are able to play a game by displaying the current board, asking the user for a move, validating this move, and applying it. The opponent is a basic agent that makes any valid move.

This is a good place to introduce the terms &lsquo;agent&rsquo;, &lsquo;state&rsquo;, and &lsquo;action&rsquo;.

At this stage it is good to nudge the students towards giving the human player and the basic AI player a common interface. This will later let us play two RL players against each other using the same framework.

## Stage 2: basic RL agent
The basic RL agent is given a state _s_, chooses an action _a_, and is given a reward _r_. _r_ = 1 if the action causes the agent to win the game, _r_ = &minus;1 if the game is lost, and _r_ = 0 otherwise. It stores a table Q: state &times; action &rarr; R and picks _a_ that maximises _Q_(_s_, _a_) for our fixed _s_.

Once we get the reward the update step is:
_Q_(_s_, _a_) := _r_ &minus; max<sub>_a_&prime;</sub>_Q_(_s_&prime;, _a_&prime;), where _s_&prime; is our new state and _a_&prime; is any possible action from _s_&prime;. If there are no possible actions, we set the min term to 0. You may recognise this as a special case of _Q_-learning where we set both the learning rate _&alpha;_ and the discount rate _&gamma;_. We replace adding max with subtracting it, because there are two agents playing against each other (this is analogous to the [Minimax algorithm](https://en.wikipedia.org/wiki/Minimax)).

The action that we choose, _a_, is the one with the highest value _Q_(_s_, _a_). We tiebreak randomly.

We should theoretically be able to train our agent by playing against it repeatedly. Note that it may take a while to train, so it&rsquo;s fairly infeasible to do by hand. So we play it off against itself.

## Stage 3: extensions
Three components that as extensions for faster students:
- We introduce _&epsilon;_-greedy and have agent choose a random action _&epsilon;_ of the time when training. This is a good time to talk about exploration vs exploitation.
- We introduce the learning rate _&alpha;_ and discount rate _&gamma;_. Our update formula then becomes _Q_(_s_, _a_) := (1 &minus; _&alpha;_) _Q_(_s_, _a_) + _&alpha;_ (_r_ &minus; _&gamma;_ max<sub>_a_&prime;</sub>Q(_s_&prime;, _a_&prime;)).
- We use symmetry to collapse equivalent states.
