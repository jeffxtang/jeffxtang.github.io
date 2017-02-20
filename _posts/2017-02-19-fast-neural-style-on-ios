---
layout: post
comments: true
title:  "How to Develop an Offline Art Filters iOS App Like Prisma"
date:   2017-02-19 13:00:00
categories: CNN, Deep Learning, AI, iOS
---

Believe it or not, yesterday, Feb 18, 2017, was the first time I heard of and tried on my iPhone the Prisma app, an AI-based offline photo app with lots of art filters, and Apple's App of the Year 2016. Sorry to disappoint you with my seemingly cave-living life (now I know that the Prisma app has been covered by "[Hundreds of Publications, With the Number Soaring Daily](http://prisma-ai.com/)") but since you found me, I'll reward you with a step-by-step guide on how to develop an iOS app just like Prisma, with regards to its deep learning and AI aspect - the iOS UI part I'd assume is easy for you.

So what did it take for me to figure out all the major deep learning related details and how to run it on iOS quickly and effectively like Prisma? It started with my review of Andrej Karpathy's [CS231n Lecture 9 on Neural Style](https://youtu.be/GHVaaHESrlY?t=52m2s) last month, and this time I decided to look into the details of how neural style gets done. So I read the original paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) published in Aug 2015, tried [Justin Johnson's Torch implementation of the algorithm](https://github.com/jcjohnson/neural-style), and [a nice TensorFlow implementation](https://github.com/anishathalye/neural-style). I also played with another simpler implementation, with good code comments, of neural style in TensorFlow, called [Neural Style Painting](https://github.com/log0/neural-style-painting).

Maybe indeed I have a deep liking for the cave-living lifestyle, or maybe it's just a calling of my ancestors hundreds of thousands of years ago - I enjoy reading [A Brief History of Humankind](https://www.amazon.com/Sapiens-Humankind-Yuval-Noah-Harari/dp/0062316095) every time time pick it - I always prefer an offline access to app features, if possible at all, so I can use the app anywhere anytime. Because the original neural style algorithm requires online optimization/training when adding style to a content image, I managed to build a TensorFlow graph file and test it on iOS to actually do the real-time training on iOS, after verifying that it is theoretically possible by following the example in this blog [Training a TensorFlow graph in C++ API](https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API). As it takes a few minutes to do the training on a modern GPU (see the final note in [my previous blog](http://jeffxtang.github.io/deep/learning,/hardware,/gpu,/performance/2017/02/14/deep-learning-machine.html) for detail), it naturally takes forever to run the train operation which minimizes the loss using AdamOptimizer on iOS - I can't stand to see my iOS simulator or device stuck in training after I'm back from dinner...

Guess I have to accept a server based solution for now, like the app from DeepArt.io - somehow I found a couple of server-based neural style iOS apps at the time? Then I don't know if it's good luck or a little more perseverance, I came across a [Fast Style Transfer in TensorFlow](https://github.com/lengstrom/fast-style-transfer) project and Justin John's paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) in March 2016 states that "Compared to the optimization-based method, our network gives similar qualitative results but is three orders of magnitude faster." Three orders of magnitude - that's like 1,000 times faster! After about a week of reading the paper and the code and debugging and dreaming and thinking about it last thing before going to sleep and first thing after waking up, I figured out how to build an iOS app that can add amazing styles to photos in a few seconds, without the need of Internet connection, and each style, after the process of training, frozen and quantized, only adds about 1.7MB to the iOS app. Here's how:

1. Get the repo at https://github.com/jeffxtang/fast-style-transfer which is a fork of Logan Engstrom's TensorFlow implementation of fast style transfer, added with the iOS offline support:
  * replaced in `transform.py` the line `preds = tf.nn.tanh(conv_t3) * 150 + 255./2` with `preds = tf.add(tf.nn.tanh(conv_t3) * 150,  255. / 2, name="preds")` so we can refer to the output result using the name `preds`;
  * added in `evaluate.py` the following lines after loading the trained checkpoint:
  ```
  saver = tf.train.Saver()
  saver.save(sess, "checkpoints_ios/fns.ckpt")
  ```
  Note that the two lines need to be added only after running `style.py` for training has been completed;
  * a Python script to freeze the trained checkpoint with input image placeholder and output image name;
  * iOS sample code that sends the input image and processes the output stylized image: the returned stylized image bitmap data gets converted to UIImage in the `tensorToBuffer` function.

2. Run `setup.sh` to download the pre-trained VGG model and the training dataset, then follow the step in "Training Style Transfer Networks" to run `style.py`, which will create the checkpoint files containing both the graph and network parameters.

3. Create a new folder named `checkpoints_ios` and uncomment the two lines of code in `evaluate.py`:
```
# before run evaluate.py, uncomment the following two lines to generate model to be frozen and deployed
# saver = tf.train.Saver()
# saver.save(sess, "checkpoints_ios/fns.ckpt")
```

4. Run `evaluate.py` as follows to generate a new checkpoint with input image placeholder and output image name:
```
python evaluate.py --checkpoint checkpoints_ios/model.ckpt \
  --in-path  examples/content/dog.jpg \
  --out-path examples/content/dog-output.jpg
```

5. Run `python freeze.py --model_folder=checkpoints_ios --output_graph fst_frozen.pb` to build a .pb file which combines the graph and the parameter values in checkpoint. This will create a file about 6.7MB.

6. Copy the `fst_frozen.pb` file to /tf_files, then in the TensorFlow source directory, run `bazel-bin/tensorflow/tools/quantization/quantize_graph --input=/tf_files/fst_frozen.pb  --output_node_names=preds --output=/tf_files/fst_frozen_quantized.pb --mode=weights`. This is the same step as step 6 of my other blog [What Kind of Dog Is It - Using TensorFlow on Mobile Device](http://jeffxtang.github.io/deep/learning,/tensorflow,/mobile,/ai/2016/09/23/mobile-tensorflow.html) and will reduce the size of the pb file to about 1.7MB, which means that for an app size of about 135MB, the size of the Prisma app, you can put in about 60 styles for offline processing (the TensorFlow iOS library takes more than 20MB size).

7. Drag and drop the `fst_frozen_quantized.pb` file to the TensorFlow sample iOS Simple project. You can move this repo's `ios_simple_fst` folder to your TensorFlow source root's `tensorflow/contrib/ios_examples` folder (because our project refers to the TensorFlow source and the library built there relatively) and launch the Xcode project and run the app on iOS simulator or device to see the effect of adding the trained style filter in the `fst_frozen_quantized.pb` to input images. Just tap on the "Run Model" button and you'll see one of the the three images included in the project stylized in about 7 seconds.

**Important** You need to enter in the iOS project the same width and height of the image used for the `img_placeholder` when running `evaluate.py`, otherwise you'll get an `Conv2DCustomBackpropInput: Size of out_backprop doesn't match computed` error when running the iOS app. To change the image width and height used in the iOS project, search for `const int wanted_width` and `const int wanted_height`, as well as `UIImage *img = [RunModelViewController convertBitmapRGBA8ToUIImage:buffer withWidth: 420 withHeight: 560];`.

There you go, you can collect many awesome-looking images for your style images and train them one by one, then freeze and quantize them and add them to your iOS app, achieving the same offline AI effect as Apple's Best iPhone App of 2016.
