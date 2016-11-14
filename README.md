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

## Weekly Progress
- [Week 3][w3]
- [Week 2][w2]
- [Week 1][w1]

[Our literature][literature]

<reference area>
 [w1]: https://docs.google.com/document/d/1s0kd8WtWGTmd1UVXTkrfDW2UO5CTqzqcIwnjeDxQePM/edit?usp=sharing
 [w2]: https://docs.google.com/document/d/1At0JQWX5_SSrzfxFpIgsBHiTF7t6Hokyu2rLuFqJhLI/edit?usp=sharing
 [w3]: https://docs.google.com/document/d/1Z7phzRq6DBAIkYaEV9gGdABzp00Eg5YjatJUpNwCUzs/edit?usp=sharing
 [literature]: https://docs.google.com/document/d/1RUpfMoQ90NK3uj2vxTj5zgKUpwJE3qYCr9V3MQ439X0/edit?usp=sharing
