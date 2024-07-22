<div align="center">
  <a href="https://github.com/VIS4ROB-lab/HyperState">
    <img src="https://drive.google.com/uc?export=view&id=1UAFr3tepqKwdnTomhKaeI2eIag3HOISY" alt="" style="width: 150px;">
  </a>

<h2><em>Hyper</em>State</h2>
  <p>
    Collection of discrete- and continuous-time motion parametrizations.
    <br />
    <a href="https://github.com/VIS4ROB-lab/HyperState/issues">Report Issues or Request Features</a>
  </p>
</div>
<br />

## About

[*Hyper*State](https://github.com/VIS4ROB-lab/HyperState) is part of
[*Hyper*SLAM](https://github.com/VIS4ROB-lab/HyperSLAM) and relies on low-level containers from
[*Hyper*Variables](https://github.com/VIS4ROB-lab/HyperVariables). In particular, *Hyper*State
implements optimization-oriented discrete- and continuous-time state parametrizations (e.g. B-Splines etc.) for motion
estimation and Simultaneous Localization Mapping pipelines. If you use this repository, please cite it as below.

```
@article{RAL2022Hug,
    author={Hug, David and B\"anninger, Philipp and Alzugaray, Ignacio and Chli, Margarita},
    journal={IEEE Robotics and Automation Letters},
    title={Continuous-Time Stereo-Inertial Odometry},
    year={2022},
    volume={7},
    number={3},
    pages={6455-6462},
    doi={10.1109/LRA.2022.3173705}
}
```
***Note:*** Development on HyperSLAM-related repositories has been discontinued.

## Installation

[*Hyper*State](https://github.com/VIS4ROB-lab/HyperState) depends on
the [Eigen](https://eigen.tuxfamily.org/), [Google Logging](https://github.com/google/glog) and
[Google Test](https://github.com/google/googletest) libraries and uses features from the
[C++20](https://en.cppreference.com/w/cpp/20) standard (see
[link](https://askubuntu.com/questions/26498/how-to-choose-the-default-gcc-and-g-version) to update gcc and g++
alternatives). The setup process itself (without additional compile flags) is as follows:

```
# Clone repository.
git clone https://github.com/VIS4ROB-lab/HyperState.git && cd HyperState/

# Run installation.
chmod +x setup.sh
sudo setup.sh

# Build repository.
mkdir build && cd build
cmake ..
make
```

## Literature

1. [Continuous-Time Stereo-Inertial Odometry, Hug et al. (2022)](https://ieeexplore.ieee.org/document/9772323)
2. [HyperSLAM: A Generic and Modular Approach to Sensor Fusion and Simultaneous<br /> Localization And Mapping in Continuous-Time, Hug and Chli (2020)](https://ieeexplore.ieee.org/document/9320417)
3. [Efficient Derivative Computation for Cumulative B-Splines on Lie Groups, Sommer et al. (2020)](https://ieeexplore.ieee.org/document/9157639)
4. [A Micro Lie Theory for State Estimation in Robotics, Solà et al. (2018)](https://arxiv.org/abs/1812.01537)
5. [A Primer on the Differential Calculus of 3D Orientations, Bloesch et al. (2016)](https://arxiv.org/abs/1606.05285)
### Known Issues and Remarks

1. Jacobians with respect to the temporal components of spline bases are not currently supported.
2. The acceleration Jacobians in SU2 have a minor bug which will be addressed in the near future.

### Updates

19.07.22 Initial release of *Hyper*State.

### Contact

Admin - [David Hug](mailto:dhug@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland  
Maintainer - [Philipp Bänninger](mailto:baephili@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland  
Maintainer - [Ignacio Alzugaray](mailto:aignacio@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland

### License

*Hyper*State is distributed under the [BSD-3-Clause License](LICENSE).
