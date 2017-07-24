---
layout: post
comments: true
title:  "Object Detection: From TensorFlow API to YOLOv2 on iOS"
date:   2017-07-23 23:50:00
categories: deep learning, CNN, object detection, tensorflow, mobile, yolo, yolov2
---

Late in May, I decided to learn more about CNN via participating in a Kaggle competition called [Sealion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count). First, I came across this nice Notebook by Radu Stoicescu: [Use keras to classify Sea Lions: 0.91 accuracy](https://www.kaggle.com/radustoicescu/use-keras-to-classify-sea-lions-0-91-accuracy), and there's a statement in it that says "This is the state of the art (object detection) method at the moment: https://cs.stanford.edu/people/karpathy/rcnn/". So I checked out Andrej Karpathy's [Playing around with RCNN, State of the Art Object Detector](https://cs.stanford.edu/people/karpathy/rcnn/) and the original RCNN paper, not realizing it was state of the art as of 2014, until days later I watched the Stanford CS231n lecture 8 video [Spatial Localization and Detection](https://www.youtube.com/watch?v=GxZrEKZfW2o&index=8&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) by Justin Johnson again (somehow the first time I watched it months ago didn't leave me any impression; maybe I just fell asleep). It's a great video and it talked about better (more state of the art, as of Feb 2016) object detection models after RCNN: Fast RCNN, Faster RCNN, and YOLO. So I spent a few more days reading the papers and looking at some github repos implementing the models. 

Then I found another algorithm called [SSD](http://www.cs.unc.edu/~wliu/papers/ssd.pdf) that claims to outperform "state-of-the-art Faster R-CNN". Well, I thought to myself, if I need to implement an object detection algorithm for the Kaggle challenge, why don't I just go with the real state of the art one? While I was learing about and working on an SSD implementation, on June 15, Google released an open source [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection). "It has to be meant for my work on the Kaggle Sealion challenge", I smiled to myself that morning when I read the news. I had almost two weeks to make it happen as the challenge's final submission deadline was June 27. 

In the following days, I was obsessed with the TensorFlow Object Detection API and managed to figure out how to train the Sealion dataset with the TF Object Detection API with a good accuracy. Below is the summary of what I did:

1. On my Ubuntu machine with Nvidia GTX 1070, I created an Anaconda environment with Keras 2 (`conda install -c conda-forge keras`) and Python 2.7;

2. I created a Python script `annotate-in-voc-xml.py`, based on Radu's notebook above, that gets the blob counts for each of the 5 sealion categories (adult males, subadult males, adult females, juveniles, and pups) and writes an XML annotation file for each image in the train set;

3. Based on the TensorFlow Object Detection API's `create_pet_tf_record.py`, and after creating a `trainval.txt` in my `/home/jeff/kaggle/SeaLions/Annotations` and `sealions_label_map.pbtxt` in `/home/jeff/ailabby/tf_object_detection/models/object_detection/data`, I wrote a script `create_sealions_tf_record.py` to generate `sealions_train.record` and `sealions_val.record` in kaggle/SeaLions;

4. After creating `faster_rcnn_resnet101_sealions.config` in `tf_object_detection/models/object_detection/samples/configs` based on `faster_rcnn_resnet101_pets.config` there, I ran the train script as follows:
```
python train.py --logtostderr --pipeline_config_path=/home/jeff/ailabby/tf_object_detection/models/object_detection/samples/configs/faster_rcnn_resnet101_sealions.config --train_dir=/home/jeff/ailabby/tf_object_detection/sealions/faster_rcnn_resnet101_coco_11_06_2017
```

This almost froze my machine. Replacing `faster_rcnn_resnet101_coco_11_06_2017` with `ssd_mobilenet_v1_coco_11_06_2017` made no difference. Neither did a small dataset of 4 images (with 4 xml files for Annotations). Looks like the image size (5616x3744) in the Sealion dataset is too big.

5. To deal with that, I tried reducing the image to 50% of the original size, and running `train.py` still almost froze the computer. 25% made the machine happy again. But the blob counts generated when running `annotate-in-voc-xml.py` on the 25% resized image were about 300 fewer than those on the original image. Maybe I could increase the blob counts by adjusting the skimage.feature.blob_log's min_sigma and max_sigma values, but the code of the "decision tree to pick the class of the blob by looking at the color in Train Dotted" in the notebook would also need to be changed as the color of the blob is not accurate anymore after resized with 25%. This method looked pretty messy.

6. Then the next morning, another idea came to my mind and I quickly tested it: by cropping every image file in Train and TrainDotted folders to 4x4 (16) subimages, I was able to run the train script successfully, and the total blob counts for the 16 subimages are pretty close to those for the original image. After an overnight training (about 15 hours) on my GTX 1070, the average loss became about 0.3. This seems to suggest that the TensorFlow Object Detection API could be used to retrain with the Kaggle Sealion dataset. 

7. Normally, running inference on a test set is much faster than training. Unfortunately, I miscalculated the time for running on the Sealion test dataset - the test set size now becomes 16 times bigger because each test image gets converted to 16 subimages. So by the deadline of the submission, I only got to complete about 20% of the test results, which had a score of 25.50170, ranked 187 out of 385 submissions. And I continued to run 3 more days and got a public score of 22.32180, which would have been ranked as 88 out of 385, a pretty mediocre result. But I'm glad I applied the TensorFlow Object Detection API to the challenge, and almost beat 200 people.

With the Kaggle Sealion competition over, I was back to my favorite topic: how to do something like this on mobile devices. That is, how can I implement the best object detection model on iOS and Android. First, after many days of dirty work with the Kaggle challenge data and experiments, I decided to give myself a nice little treat by rewatching Andrej's [Deep Learning for Computer Vision video in Deep Learning School 2016](https://www.youtube.com/watch?v=u6aEYuemt0M&index=2&list=PLrAXtmErZgOfMuxkACrYnD2fTgbzk2THW) and there I found YOLO is one of his favorite object detectors. So I did a more careful look at it and to my surprise, or I should say not surprisingly, the v2 of YOYO, aka [YOLO 9000](https://pjreddie.com/darknet/yolo/) claims on Dec 25, 2016 to, again, outperform "state-of-the-art methods like Faster R-CNN with ResNet and SSD". This seems a little crazy. Guess I should learn to expect this kind of craziness from now on.

So I checked out a nice Github repo [Darkflow](https://github.com/thtrieu/darkflow/), the TensorFlow port of Darknet, an open source neural network framework on which the original YOLO v1 and v2 implementation were based. The repo has nice documentation on how to build Tensorflow models for YOLO v1 and v2, and suggests that the `output` tensor can just be used on iOS for post processing. The TensorFlow Android examples actually also have a good implementation of object detection using the tiny-yolo model. It took me quite a few days of reading the YOLO v1 and v2 papers, debugging the Darkflow code and and the Tensorflow Android TF-Detect example to get the iOS example code for image preprocessing and post processing done correctly so I can get a stand-alone YOLO v2 model running on iOS - the actual device, not just the simulator. Here's my repo [yolov2-tf-ios](https://github.com/jeffxtang/yolov2_tf_ios) which runs the tiny YOLO v2 model nicely on iPhone (the larger YOLO v2 model runs on simulator but crashes on iPhone) and shows the detected objects with bounding boxes. Based on the repo, I built an iOS app that lets users select a photo or take a picture to see what objects are in the picture and where they are. Its accuracy still needs furture improvement, but I'd take it as a closure of my 8-week effort on object detection, or another small milestone of my AI journey of a thousand miles. It's only going to get more exciting next.













