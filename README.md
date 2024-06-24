# Soft Computing

This repository contains three projects from the Soft Computing course, organized into three folders: `Soft K1`, `Soft K2`, and `Soft K3`.

## Repository Structure

- **K1**
  - `K1.pdf`: Task description.
  - `K1/`: Folder containing the solution.
    - `SC23-G5-RA111-2020.py`: Python script with the solution.
    - picture data in two folder(pictures1 and pictures2).
    - `tester.py` test script for calculation error(MAE error)
    - `bulbasaur_count.csv` and `squirtle_count.csv` file with accurate results
    - `requirements_win.txt`

- **K2**
  - `K2.pdf`: Task description.
  - `K2/`: Folder containing the solution.
    - `SC23-G5-RA111-2020.py`: Python script with the solution.
    - `tester.py` test script for calculation error(MAE error)
    - data1 and data2 folder with test video and pictures for training

- **K3**
  - `K3.pdf`: Task description.
  - `K3/`: Folder containing the solution.
    - `SC23-G5-RA111-2020.py`: Python script with the solution.
    - `tester.py` test script for calculation error
    - data1 and data2 folder with test pictures

## Installation

In the `K1` folder, there is a `requirements_win.txt` file that lists the libraries required for the solution. To install these libraries, use the following command:

```bash
pip install -r requirements_win.txt
```

## Project Descriptions

### K1: Image Analysis and Processing with Segmentation

The first project in `K1` focuses on image analysis and processing, where the goal is to count the number of Pokemon present in the test images.

- **Task**: Analyze and process images to count the number of Pokémon.
- **Solution**: Implemented in the `SC23-G5-RA111-2020.py` script located in the `K1` folder.
- **Run the script**: To execute the solution script, use the following command in the terminal.
  ```bash
  python SC23-G5-RA-111-2020.py pictures2/
  ```

### K2: Video Analysis using HOG Image Descriptor

The second project in `K2` focuses on video analysis using the Histogram of Oriented Gradients (HOG) image descriptor. The goal is to detect collisions between cars and a red line in the videos.

- **Task**: Process and analyze videos frame by frame to detect collisions with the red line.
- **Solution**: Implemented in the `SC23-G5-RA111-2020.py` script located in the `K2` folder.
- **Run the script**: To execute the solution script, use the following command in the terminal. Processing is time-consuming due to approximately 5 videos, each with about 100 frames:

  ```bash
  python SC23-G5-RA-111-2020.py data2/

### K3: OCR Implementation

The third project in `K3` focuses on Optical Character Recognition (OCR) implementation. The task involves training and testing on a dataset of images containing Czech language characters from the `data2` folder.

- **Task**: Implement OCR to recognize Czech language characters from images.
- **Solution**: Implemented in the `SC23-G5-RA111-2020.py` script located in the `K3` folder.
- **Run the script**: To execute the OCR script, use the following command in the terminal:

  ```bash
  python SC23-G5-RA111-2020.py data2/
