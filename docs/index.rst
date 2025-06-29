.. Angstrom documentation master file, created by
   sphinx-quickstart on Tue Mar 19 10:00:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Angstrom's documentation!
====================================

Angstrom is a Python library for phase-based motion amplification in videos. It uses complex steerable pyramids to decompose video frames and amplify subtle motion by manipulating phase coefficients.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   python_package
   modules

Installation
------------

Install Angstrom using pip:

.. code-block:: bash

   pip install angstrom

Quick Start
-----------

Here's a simple example of how to use Angstrom for motion amplification:

.. code-block:: python

   from angstrom.core.motion_amplifier import MotionAmplifier

   # Initialize the motion amplifier
   amplifier = MotionAmplifier()

   # Process a video with motion amplification
   amplifier.process_video(
       input_path="input_video.mp4",
       output_path="amplified_video.mp4",
       amplification_factor=10,
       frequency_range=(0.1, 2.0)  # Hz
   )

Features
--------

* **Phase-based motion amplification**: Uses complex steerable pyramids for accurate motion detection
* **Temporal filtering**: Apply bandpass filters to target specific motion frequencies
* **GPU acceleration**: Leverages PyTorch for efficient computation
* **Multiple output formats**: Support for various video formats
* **Configurable parameters**: Fine-tune amplification factors and frequency ranges

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
