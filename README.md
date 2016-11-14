# Robotics Project Plan - Look at this
by Kevin Kepp and Philipp Braunhart
supervising tutors: Manuel Baum and Roberto Martín-Martín

## Description

Your task is to make a robot find and look at target image patches in  its field of view. The robot is supposed to solve this problem by learning a direct mapping from images to the movement of its neck-joints. This mapping should be implemented as a neural network and you can explore supervised and reinforcement learning methods to train it. If you finish quickly, we can possibly extend this project to a camera that is mounted on a robot arm, giving you way more degrees of freedom to control. (Background: computer vision, machine learning, c/c++ or python).

#### Problem Definition
Input: Images
Desired location of the image we want to center (we will change this by the activation of another DNN that detects the patch we want to find)

Output:
Motor commands (2 dimensional)

## Milestones

<a name="1"></a>
1. Make a solid plan 
  1. plan together
  2. discuss plan with Manuel and Roberto

<a name="2"></a>
2. Get Acquainted with the material
  1. read some papers (later some more)
  2. install python frameworks (caffee,lasagne,...) and test some toyexamples
  3. do tutorials for further knowledge
  4. gather first ideas to solve the problem

<a name="3"></a>
3. Write first simple simulator (should be extendable from 3x3 to more)
  1. agree on structure of the simulator so both know how to use it (class,methods,...)
  2. actually write it
  3. generate ground truth
  4. test it with simple tests

<a name="4"></a>
4. apply first small RL & perhaps NN to simulation (seperately or as one)
  1. formulate/take given reward fct.
  2. try existing architectures (e.g. last layer of deep-q [paper][paper-atari])

<a name="5"></a>
5. design own architectures and reward fct
  1. think of own architectures or ways to improve existing ones (justify why to choose this)
  2. think about applying knowledge from the papers ([learning state representations][paper-learn-srep])

<a name="6"></a>
6. apply and test own architecture and compare it to others
  1. apply our designed models on toy examples to verify
  2. apply to bigger problems (bigger image in simulator)
  3. compare performance in comparison to other methods

## Weekly Progress
- [w3][Week 3]
- [w2][Week 2]
- [w1][Week 1]


[Our literature][literature]

<reference area>
 [w1]: https://docs.google.com/document/d/1s0kd8WtWGTmd1UVXTkrfDW2UO5CTqzqcIwnjeDxQePM/edit?usp=sharing
 [w2]: https://docs.google.com/document/d/1At0JQWX5_SSrzfxFpIgsBHiTF7t6Hokyu2rLuFqJhLI/edit?usp=sharing
 [w3]: https://docs.google.com/document/d/1Z7phzRq6DBAIkYaEV9gGdABzp00Eg5YjatJUpNwCUzs/edit?usp=sharing
 [literature]: https://docs.google.com/document/d/1RUpfMoQ90NK3uj2vxTj5zgKUpwJE3qYCr9V3MQ439X0/edit?usp=sharing
