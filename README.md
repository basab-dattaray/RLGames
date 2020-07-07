Reinforcement Learning - Basab Dattaray
Running the Demos
Demos are available in the sub-package ws/demos. Steps:

Download from the Github url: https://github.com/basab-dattaray/RLGames.git.
To install the supporting python packages run: pip3 install -r requirements.txt
Navigate to the relative path "ws/Demos".
By exploring through the sub folders here, you will be able to find demos using different strategies of reinforcement learning such as planning (model based), model_free, function_approximations and policy_gradient. The strategy and other attributes for a demo is defined in the demo_APP_INFO.JSON file that exists in the same folder as the demo. Please note that strategy sub-classifications are nested by using a dot (.) nomenclature. For example an demo_APP_INFO.JSON file that accompanies a demo includes the entry "STRATEGY": "model_based.policy_gradient.ppo" implies an the demo uses an agent category "ppo" which is a subcategory of "policy_gradient" category, which is in turn a subcategory of the "model_free" category. This classification by hierarchical strategies enables efficient module sharing across like-strategies.

The agent implementations could use either PyTorch or Keras/Tensorflow deep learning libraries. Do note that any python file prefixed with the word "demo" (e.g. demo_run, demo_train or demo_test) are python applications that can be run.

Summary
This Github site was inspired by code samples from various RL algorithms. Here, reinforcement learning (RL) concepts from various quarters were reconstructed in a common framework to enable the assimilation of various aspects of reinforcement learning. This framework enables one to focus on the essential algorithms without having to understand the common boilerplate code anew for each demo/app.

The primary objective for this framework is to introduce modularity and configurability while reducing coupling across cohesive packages/modules. At the highest level of abstraction, the agent and environment are cleanly separated into their own packages.

The ultimate hope is that all of us can learn and contribute towards a better understanding of the plethora of basic theories and techniques that have been established and continue to be developed by pioneers in the dynamic and exciting field of Reinforcement Learning (RL). Perhaps a framework like this will enable practitioners in the RL field to understand the tradeoffs in picking different approaches to solve problems.

References:
Shown below are just a few of the Reinforcement Learning (RL) training material that helped me understand key RL concepts. There are many other good reference materials on the subject: more references could be added as other suggestions are received.

Reinforcement Learning: An Introduction - Book, 2nd Edition:

Richard Sutton Tutorial: Introduction to Reinforcement Learning with Function Approximation - YouTube:

RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning - YouTube:

Balaraman Ravindran, NPTEL :: Computer Science and Engineering - NOC:Reinforcement Learning - YouTube

John Schulman 1: Deep Reinforcement Learning - YouTube

Hado Van Hasselt: Reinforcement Learning 1: Introduction to Reinforcement Learning - YouTube

Peter Abeel, Andrej Karpathy: Deep RL Bootcamp 2017 - YouTube

John Schulman : Deep RL Bootcamp Lecture 5: Natural Actor Gradients, TRPO, PPO - YouTube

Sergey Levine : Deep Reinforcement Library, CS 294-112 at UC Berkeley - YouTube

Lex Fridman: MIT 6.S091: Introduction to Deep Reinforcement Learning (Deep RL) - YouTube

Skowster the Geek: Proximal Actor Optimization (PPO) Tutorial - Master Roboschool - YouTube