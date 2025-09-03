# Reinforcement Learning (RL) - 100 Interview Questions & Answers

---

### Q1. What is Reinforcement Learning (RL)?
A learning paradigm where an agent interacts with an environment, takes actions, and learns from rewards/punishments.

---

### Q2. What are the key components of RL?
- Agent  
- Environment  
- State (s)  
- Action (a)  
- Reward (r)  
- Policy (π)  
- Value functions  

---

### Q3. What is a policy in RL?
A mapping from states to actions (deterministic or stochastic).

---

### Q4. What is a reward in RL?
A scalar signal from the environment indicating the quality of an action.

---

### Q5. What is a return in RL?
The cumulative discounted sum of rewards:  
**G_t = Σ γ^k r_(t+k+1)**

---

### Q6. What is a value function?
A function estimating expected return from a state or state-action pair.

---

### Q7. What is a Q-value?
The expected return for taking action a in state s under policy π.

---

### Q8. What is the Bellman equation?
Recursive relation describing value of a state:  
**V(s) = E[r + γ V(s’)]**

---

### Q9. What is a Markov Decision Process (MDP)?
A framework for RL defined by (S, A, P, R, γ).

---

### Q10. What is the Markov property?
The next state depends only on the current state and action, not history.

---

### Q11. What is a discount factor (γ)?
A value (0 ≤ γ < 1) that reduces the importance of future rewards.

---

### Q12. What is the exploration-exploitation tradeoff?
Balancing between exploring new actions and exploiting known good actions.

---

### Q13. What is an episodic task?
An RL task with a clear start and end (e.g., a game).

---

### Q14. What is a continuing task?
An RL task without a natural terminal state (e.g., stock trading).

---

### Q15. What is the difference between on-policy and off-policy RL?
- On-policy: learns about the policy being executed (e.g., SARSA).  
- Off-policy: learns about a different policy (e.g., Q-learning).  

---

### Q16. What is model-free RL?
Methods that learn without an explicit model of the environment.

---

### Q17. What is model-based RL?
Methods that use a model of environment dynamics for planning.

---

### Q18. What is dynamic programming in RL?
A family of algorithms solving MDPs when the model is fully known.

---

### Q19. What is Monte Carlo learning?
Learning from complete episodes by averaging returns.

---

### Q20. What is Temporal Difference (TD) learning?
Updating estimates using bootstrapping:  
**V(s) ← V(s) + α [r + γ V(s’) – V(s)]**

---

### Q21. What is SARSA?
An on-policy TD control algorithm:  
**Q(s,a) ← Q(s,a) + α [r + γ Q(s’,a’) – Q(s,a)]**

---

### Q22. What is Q-learning?
An off-policy TD algorithm:  
**Q(s,a) ← Q(s,a) + α [r + γ max_a’ Q(s’,a’) – Q(s,a)]**

---

### Q23. What is the difference between SARSA and Q-learning?
- SARSA: on-policy, learns the policy being followed.  
- Q-learning: off-policy, learns the optimal policy independently.  

---

### Q24. What is policy iteration?
Iteratively evaluating and improving a policy until optimal.

---

### Q25. What is value iteration?
Combining policy evaluation and improvement into a single update.

---

### Q26. What is epsilon-greedy strategy?
With probability ε, explore random action; otherwise, exploit best known.

---

### Q27. What is softmax action selection?
Choosing actions probabilistically based on their Q-values.

---

### Q28. What is actor-critic in RL?
Combines policy learning (actor) with value learning (critic).

---

### Q29. What is REINFORCE algorithm?
A Monte Carlo policy gradient method updating weights via sampled returns.

---

### Q30. What is a policy gradient?
Optimization method updating policy parameters directly using gradient of expected return.

---

### Q31. What is the advantage function?
A(s,a) = Q(s,a) – V(s)  
Measures how much better an action is compared to average.

---

### Q32. What is entropy regularization?
Adding entropy to the objective to encourage exploration.

---

### Q33. What is DQN?
Deep Q-Network — uses deep neural nets to approximate Q-values.

---

### Q34. What are key innovations in DQN?
- Experience replay  
- Target network  
- CNNs for state representation  

---

### Q35. What is experience replay?
Storing transitions and sampling them randomly to break correlations.

---

### Q36. What is a target network in DQN?
A separate network updated slowly for stable Q-learning targets.

---

### Q37. What is Double DQN?
Uses two networks to reduce overestimation bias in Q-learning.

---

### Q38. What is Dueling DQN?
Splits network into value and advantage streams for efficiency.

---

### Q39. What is Prioritized Experience Replay?
Sampling important transitions more frequently based on TD error.

---

### Q40. What is Rainbow DQN?
An integration of multiple improvements: Double DQN, dueling, PER, etc.

---

### Q41. What is Deep Deterministic Policy Gradient (DDPG)?
An actor-critic algorithm for continuous action spaces.

---

### Q42. What is Twin Delayed DDPG (TD3)?
Improved DDPG with twin critics and delayed actor updates.

---

### Q43. What is Soft Actor-Critic (SAC)?
An off-policy algorithm maximizing both reward and entropy.

---

### Q44. What is PPO (Proximal Policy Optimization)?
A policy gradient algorithm using clipped surrogate objective for stability.

---

### Q45. What is TRPO (Trust Region Policy Optimization)?
Uses KL-divergence constraint to update policies conservatively.

---

### Q46. What is A3C?
Asynchronous Advantage Actor-Critic — multiple agents update in parallel.

---

### Q47. What is A2C?
Synchronous version of A3C (Advantage Actor-Critic).

---

### Q48. What is imitation learning?
Learning a policy by mimicking expert demonstrations.

---

### Q49. What is inverse reinforcement learning (IRL)?
Inferring the reward function from observed behavior.

---

### Q50. What is apprenticeship learning?
Learning optimal policies by combining IRL and RL.

---

### Q51. What is hierarchical RL?
Decomposing tasks into subgoals and subpolicies.

---

### Q52. What is option framework in RL?
Extends actions to include temporally extended options.

---

### Q53. What is curriculum learning?
Training agents on progressively harder tasks.

---

### Q54. What is multi-agent RL?
Multiple agents learning and interacting in the same environment.

---

### Q55. What is cooperative multi-agent RL?
Agents collaborate to maximize a shared reward.

---

### Q56. What is competitive multi-agent RL?
Agents compete with conflicting objectives.

---

### Q57. What is self-play in RL?
An agent trains by playing against copies of itself.

---

### Q58. What is AlphaGo?
A breakthrough RL system combining deep learning, MCTS, and self-play.

---

### Q59. What is Monte Carlo Tree Search (MCTS)?
A planning algorithm combining random rollouts with tree search.

---

### Q60. What is AlphaZero?
General RL algorithm using self-play + MCTS for multiple games.

---

### Q61. What is MuZero?
A model-based RL method that learns dynamics without knowing rules.

---

### Q62. What is World Models?
RL agents learning latent dynamics of the environment.

---

### Q63. What is model predictive control (MPC)?
Planning by optimizing actions over a finite horizon using a model.

---

### Q64. What is reward shaping?
Adding extra signals to accelerate learning.

---

### Q65. What is sparse reward problem?
When rewards are rare, making exploration difficult.

---

### Q66. What is dense reward?
Frequent feedback guiding learning step-by-step.

---

### Q67. What is intrinsic motivation in RL?
Using curiosity or novelty bonuses to drive exploration.

---

### Q68. What is curiosity-driven learning?
Agents rewarded for predicting and reducing surprise.

---

### Q69. What is exploration bonus?
Extra reward encouraging exploration of unseen states.

---

### Q70. What is reward hacking?
When agent exploits loopholes in reward function without solving task.

---

### Q71. What is generalization in RL?
Ability to perform well on unseen states/environments.

---

### Q72. What is transfer learning in RL?
Using knowledge from one task to accelerate learning in another.

---

### Q73. What is meta-RL?
Training agents to quickly adapt to new tasks.

---

### Q74. What is offline RL?
Learning solely from pre-collected datasets.

---

### Q75. What is batch RL?
Similar to offline RL — learning from fixed datasets without exploration.

---

### Q76. What is online RL?
Learning while actively interacting with environment.

---

### Q77. What is continuous action space RL?
Agents choose from infinite continuous actions (e.g., robotics).

---

### Q78. What is discrete action space RL?
Agents select from a finite set of actions.

---

### Q79. What is hybrid action space?
Combination of discrete and continuous actions.

---

### Q80. What is exploration vs exploitation dilemma?
Whether to try new actions or leverage best known action.

---

### Q81. What is bootstrapping in RL?
Using estimates of future returns to update current value.

---

### Q82. What is eligibility trace?
Mechanism combining MC and TD methods (TD(λ)).

---

### Q83. What is TD(λ)?
Generalization of TD methods with eligibility traces.

---

### Q84. What is n-step return?
Extending TD by using n-step lookahead before bootstrapping.

---

### Q85. What is policy evaluation?
Estimating the value of a policy.

---

### Q86. What is policy improvement?
Improving policy based on updated value estimates.

---

### Q87. What is policy gradient theorem?
Provides formula to compute gradients for policy optimization.

---

### Q88. What is advantage actor-critic?
Actor uses advantage estimates instead of raw returns.

---

### Q89. What is importance sampling in RL?
Technique for off-policy corrections in policy gradient methods.

---

### Q90. What is KL-divergence in RL optimization?
A measure of distance between old and new policies for stability.

---

### Q91. What is catastrophic forgetting in RL?
When agents forget old knowledge while learning new tasks.

---

### Q92. What is stability-plasticity dilemma?
Tradeoff between retaining old knowledge and learning new.

---

### Q93. What is safe RL?
Ensuring agents avoid unsafe or harmful actions.

---

### Q94. What is constrained RL?
RL with explicit constraints (e.g., resource or safety limits).

---

### Q95. What is multi-objective RL?
RL where agent optimizes multiple conflicting objectives.

---

### Q96. What is explainable RL?
Making RL policies interpretable to humans.

---

### Q97. What is sample efficiency?
How much data an RL algorithm needs to learn effectively.

---

### Q98. What is computational efficiency?
How much compute an RL algorithm requires.

---

### Q99. What are real-world challenges in RL?
- Sparse rewards  
- Safety  
- Sample inefficiency  
- Generalization  
- Partial observability  

---

### Q100. What are major applications of RL?
- Robotics  
- Games (AlphaGo, Dota2, Chess)  
- Finance  
- Healthcare  
- Autonomous driving  
- Recommendation systems  

---
