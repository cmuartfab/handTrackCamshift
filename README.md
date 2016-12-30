# Introduction

This repository contains code for using computer vision to track the human hand.  The code is built on OpenCV and written in Python.

# Usage

There are two options available:

- Run the code from the command window without any input arguments;  This way we will detect the hand first and once detected, we will track it through the video.  

`python HaarHandTracker.py`
     
- Run the code from the command window with some random input argument, the value of the argument can be anything but there has to be an argument.  The program will ask you to place your hand in a window and this would be used to track further. There will not be any detection in this case. This has not been tested completely yet.  

`python HaarHandTracker.py arg1`
      
