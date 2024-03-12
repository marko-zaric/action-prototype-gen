# Effect-driven Motion Space Discretization

[![Python 3.8.10+](https://img.shields.io/badge/python-3.8.10+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3810/)

This package provides the implementation of the effect-centric motion/action space discretization approach presented in the paper Unsupervised Learning of Effective Actions in Robotics. It can be used with any continuous motion/action space and will generate discrete action prototypes each producing different effects in the environment. After an exploration phase, the algorithm automatically builds a representation of the effects and groups motions into action prototypes, where motions more likely to produce an effect are represented more than those that lead to negligible changes.

## Installation

To use this package you can simply use pip to directly install this package:

`pip install git+https://github.com/marko-zaric/action-prototype-gen.git`

## Usage
