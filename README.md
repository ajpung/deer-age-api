## Introduction

This repository contains scripts used to predict deer age via Machine Learning and Computer Vision. In the approach, two separate models are developed -- one for age prediction via trail camera images, and another using jawbone images. Each technique a different model, based on multi-fold CNN ensembles.

In an effort to make all models fast, efficient, and streamlined, models are built to accept both grayscale and colored images, and utilize only a single optimized model. Furthermore, color and grayscale images are handled together in the modeling pipeline. 
