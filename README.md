# Robotics Project Plan - Look at this
by Kevin Kepp and Philipp Braunhart

## Monday agenda

- shortly present the problem
- what is our plan

## Current TODOs
| TODOs | Kevin | Philipp|
|:---|:---:|:---:|
|making project plan|X|X|
|discussing plan with Roberto & Manuel|X|X|
|setting up github|O|O|
|setting up weekly live google doc|O|O|
|look at q-learning (wiki)|O|O|
|[reading paper saccadic][paper-saccadic]|-|O|
|[reading paper ent-to-end][paper-end2end]|O|-|
|[reading paper learning state rep][paper-learn-srep]|-|O|
|[reading paper deepmind atari][paper-atari]|O|O|
|install theano, lasagne, caffe-recipes and test them|O|O|
|[deeplearning tutorial][tutorial-dl]|O|O|
|[Reinf. Learn tutorial][tutorial-rf]|O|O|
|[lasagne tutorial][tutorial-lasagne]|O|O|
|[getting to know caffe modelZoo][caffe-modelzoo]|O|O|
|[extract caffe model to lasagne][howto-caffe-model-to-lasagne]|O|O|
|think about mvp|O|O|

<for easy usage:
|nicetext|O|O|>

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

- either position or velocities for motor command unit
- 1st design simple (without ros, then with ros) (big image - cut out region)
- 2nd design it close to the real system (limits, motor commands, image)

<a name="4"></a>
4. apply first small RL & perhaps NN to simulation (seperately or as one)
  1. formulate/take given reward fct.
  2. try existing architectures

<a name="5"></a>
5. design own architectures and reward fct
  1. think of own architectures or ways to improve existing ones (justify why to choose this)
  2. think about applying knowledge from the papers ([learning state representations][paper-learn-srep])

<a name="6"></a>
6. apply and test own architecture and compare it to others
  1. apply our designed models on toy examples to verify
  2. apply to bigger problems (bigger image in simulator)
  3. compare performance in comparison to other methods

<a name="7"></a>
7. get out of simulation into the real game
  1. get to know the apis of the camera and robot control
  2. Test apis (get image, move camera/robot)
  3. Be able to process camera input automatically in a meaningful way

<a name="8"></a>
8. Robot is not moving, but camera focuses on some given object
  1. apply methods (ours and given) on some easy object (e.g. white paper with big red dot) and compare results (think of reasons why one is better)
  2. apply on some sophisticated object (e.g. face) and compare results (think of reasons why one is better)

<a name="9"></a>
9. Robot moves through lab and focuses view on some given object
  1 use better method to implement with movement of robot through the lab (adding more output neurons)
  2. test the limits of this

optional:

Z. Extend it to make the robot learn how to get the most information out of surrounding of object

## Timeline

|weeks|until|milestone|outcome (problems, solutions)|
|:---|:---|---|---|
|1|31.10.|[1](#1) & [2](#2) | |
|2|07.11.| [3](#3) | |
|3|14.11.| [4](#4) | |
|4|21.11.| [5](#5) | |
|5|28.11.| [6](#6) | |
|6|05.12.| [7](#7) | |
|7|12.12.| | |
|8|19.12.| vacation | |
|9|26.12.| vacation | |
|10|02.01.| [8](#8) | |
|11|09.01.| [9](#9) | |
|12|16.01.| | |
|13|23.01.| | |
|14|30.01.| | |
|15|06.02.| | |
|16|13.02.| | |
|16|20.02.| | |

<reference area>
 [paper-end2end]: https://arxiv.org/abs/1504.00702
 [paper-learn-srep]: http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf
 [paper-saccadic]: https://arxiv.org/pdf/1610.06492v1.pdf
 [paper-atari]: https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
 [tutorial-rf]: http://karpathy.github.io/2016/05/31/rl/
 [tutorial-lasagne]: http://lasagne.readthedocs.io/en/latest/user/tutorial.html
 [tutorial-dl]: http://deeplearning.net/tutorial/
 [howto-caffe-model-to-lasagne]: https://github.com/Lasagne/Recipes/blob/master/examples/Using%20a%20Caffe%20Pretrained%20Network%20-%20CIFAR10.ipynb
 [caffe-modelzoo]: https://github.com/BVLC/caffe/wiki/Model-Zoo

## cheatsheet area

|TODO signals| description|
|:---:|---|
| O | open |
| X | done |
| - | not for me|
