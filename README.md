# 3d-memory

## Preparation

### Set up the hand_object_detector submodule

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
```

Download the [faster_rcnn_1_8_132028.pth](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) and **Save model** in the **models/** folder:
```
hand_object_detector
└── models
    └── res101_handobj_100K
        └── pascal_voc
            └── faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth
```

## Start the app

First, start the frame synchornizer which synchornizes the rgb frame, depth frame and detic result by:
```
python frame_synchornize.py run
```

Then, start the 3d memory by:
```
python 3d_memory.py run
```
