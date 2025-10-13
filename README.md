## Introduction

This repository contains scripts used to predict deer age via Machine Learning and Computer Vision. In the approach, two separate models are developed: one for age prediction via trail camera images, and a second using jawbone images. Each technique utilizes a different model.

In an effort to make all models more efficient, each one is built to accept both grayscale and colored images  and utilize only a single optimized model (no ensembles). Augmentation further helps develop the model, applying a series of small changes to the base training image set.

Updates:
- 10/3/25: Latest model installed for trail camera imagery, utilizing both color and grayscale images.

-10/1/25: Exploration into other factors (location, image datetime) completed; not enough data to be useful, even when expanding data into multiple agencies.