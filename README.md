# Voice Conversion with Non-Parallel Data
## Using your own dataset


## How to make it work:
First, I set up a docker container with this docker container `run -dit -v /mas:/mas -v /u:/u -v /dtmp:/dtmp -v /tmp:/tmp --name $(whoami)-deep-voice-changer tensorflow/tensorflow:latest-gpu`. I did everything specified by Necsys plus these aditional steps:

1. apt-get install libsndfile1
2. apt-get install ffmpeg libavcodec-extra
3. apt-get install build-essential libcap-dev
4. apt-get install nano
5. apt-get install screen
6. apt-get install virtualenv (or python-virtualenv)
7. apt-get install git

After that, I sudo'd and cloned the repository by andabi in /u/$(whoami). Then, make an environment and pip install following packages: 

* librosa == 0.6.2 (make sure that numba == 0.48 and llvmlite==0.33.0)
* tensorflow-gpu==1.15 (makse sure you pip uninstall tensorflow before, seems to help)
* tensorpack==0.9.0.1
* pydub
* soundfile
* git+https://github.com/wookayin/tensorflow-plot.git@master
* joblib
* pyyaml
* python-prctl
* gdown

Once you have these packages, you have to modify two lines in the tensorpack file. Do `nano $(nameOfEnv)/lib/python3.6/site-packages/tensorpack/graph_builder/utils.py` and change "from tensorflow.contrib import nccl" to "from tensorflow.python.ops.nccl_ops import all_sum" and change "summed = nccl.all_sum(grads)" to "summed = all_sum(grads)". These lines are in the function allreduce_grads. 

Finished that? Good, almost there! Now do `nano hparams/default.yaml` and change "logdir_path: '/u/$(WHOAMI)/deep-voice-conversion/training'" (remember to make the folder 'training'). Then, in the same thing change `data_path: '/u/$(WHOAMI)/deep-voice-conversion/datasets/arctic/slt/*.wav'`, which should be under train2. 

One last thing is missing, you have to install the pretrained model for train1. Run the following commands:

* `gdown https://drive.google.com/uc?id=1yC3G3V03X3s8mKJ1J6bMkOqDT8r-TBb8`
* `python`
* In the python terminal thing, write:


    `import zipfile`
    `with zipfile.ZipFile("train1.zip", 'r') as zip_ref:`
        `zip_ref.extractall("training")`
