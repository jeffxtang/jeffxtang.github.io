---
layout: post
comments: true
title:  "What Kind of Dog Is It - Using TensorFlow on Mobile Device"
date:   2016-09-23 12:07:57
categories: deep learning, tensorflow, mobile, AI
---

Even before I had my first dog, a Labrador Retriever, in June 2015, while walking and seeing a dog I often wondered what kind of breed it is. About a year ago, I found the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and then asked a friend who had a Ph.D. in computer vision from CMU if it's possible to use it and some machine learning algorithm on my iPhone and reach a recognition precision of about 80% or 90%, and this is what he told me: "80-90% will be really hard, unless you are willing to restrict your problem in some way. For example, request a user to take a picture in some specific angle, or reduce the number of classes." Also he said that "for deep learning to work, you will need a lot more data (than the Stanford Dogs Dataset, which has about 100-200 images for each dog breed) to train the neural network".

Convinced naturally but disillusioned, I devoted more time and love for my own Lab dog. Late 2015, Google released their open-source AI and machine learning framework [TensorFlow](https://www.tensorflow.org) and I played with it for a while. Then at Google I/O in May 2016, I attended the session [Machine Learning: Google's Vision](https://www.youtube.com/watch?v=Rnm83GqgqPE) and found this amazing codelab [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets). Basically, I was informed that I can use TensorFlow and Google's publicly released deep learning model for image classification to retrain the model on a new image classification task of my own so I can perform my own image classification. And Google claims that TensorFlow can run on different platforms including mobile devices such as iPhone and Android phones. Wasn't that exactly what I tried to do?

It turns out that following the steps in the codelab above and then replacing the images with the Stanford Dogs Dataset to classify a dog on a command line in my computer is pretty straightforward, but it takes a lot more effort to finally be able to successfully run the classification on my iPhone. Below is the step-by-step summary of the whole process I went through:

1. Follow the steps in the codelab [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets) to install TensorFlow and see how a sample retraining works. I also did a [Pip installation](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation) of TensorFlow using TensorFlow 0.8.0 binary (`https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl`) and TensorFlow 0.10.0rc0 binary (`https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl`) respectively on my two Mac's by setting `export TF_BINARY_URL=` before running `sudo pip install --upgrade  $TF_BINARY_URL`. I realized that this might not be the best way to set up TensorFlow but it worked fine for me. Note that I had to keep the older 0.8.0 binary because a script named strip_unused, needed to run to fix a runtime error in a retrained model on iOS, works only on 0.8.0 and breaks after 0.8.0 (up till at least 0.10.0rc0). Note also that the latest TensorFlow release as of this writing is 0.10.0, which may have fixed the bug in running strip_unused after 0.8.0. [**Update 09/24/2016**: The latest source release v0.10.0, available [here](https://github.com/tensorflow/tensorflow/releases) has indeed fixed the strip_unused issue. So now you can just install the TensorFlow 0.10.0 binary along with its latest source to build the strip_unused and run it (see step 5 below).]

2. Download the Google's [Inception v3 model](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) and the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). I unzipped the Inception v3 zip file and moved it to `/tf_files` and unzipped the dog dataset to `~/Downloads/dog_images`.

3. Get the TensorFlow source and build for iOS samples:
```
cd
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
tensorflow/contrib/makefile/build_all_ios.sh
```
You may also get a specific, relatively more stable, release of the TensorFlow source, such as v0.10.0,  [here](https://github.com/tensorflow/tensorflow/releases).

4. From the root of the TensorFlow source, build and then run the retrain script:
```
bazel build tensorflow/examples/image_retraining:retrain
bazel-bin/tensorflow/examples/image_retraining/retrain \
  --model_dir=/tf_files/inception-v3 \
  --output_graph=/tf_files/retrained_models/dog_retrained.pb \
  --output_labels=/tf_files/retrained_models/dog_retrained_labels.txt \
  --image_dir ~/Downloads/dog_images \
  --bottleneck_dir=/tf_files/dogs_bottleneck
```
You may need to run `./configure` from the TensorFlow source root first before running `bazel build ...`. After this step, you can build and run the `label_image` script, documented [here](https://www.tensorflow.org/versions/r0.10/how_tos/image_retraining/index.html) to verify that the top 5 accuracy of the dog breed classification is pretty high - in my test it's about 90%.

5. Build and run the strip_ununsed script:
```
bazel build tensorflow/python/tools:strip_unused
bazel-bin/tensorflow/python/tools/strip_unused \
  --input_graph=/tf_files/retrained_models/dog_retrained.pb \
  --output_graph=/tf_files/retrained_models/stripped_dog_retrained.pb \
  --input_node_names=Mul \
  --output_node_names=final_result \
  --input_binary=true
```
This is needed to fix a DecodeJpeg issue caused by the retraining model. See [this github issue](https://github.com/tensorflow/tensorflow/issues/2883) and [another one](https://github.com/tensorflow/tensorflow/issues/3480) for more details.

6. Build and run the quantize_graph script:
```
bazel build tensorflow/contrib/quantization/tools:quantize_graph
bazel-bin/tensorflow/contrib/quantization/tools/quantize_graph \    
  --input=/tf_files/retrained_models/stripped_dog_retrained.pb \
  --output_node_names=final_result \
  --output=/tf_files/retrained_models/quantized_stripped_dogs_retrained.pb \
  --mode=weights
```
This step is needed to make the model run successfully on iOS. Before quantization, the retrained model size is almost 90MB, and the app would just crash when the model is loaded on an actual iOS device. With quantization, the model size is only a little over 20MB. For more details on why and how quantization works, see [Pete's blog](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/) or TensorFlow's [How To Quantize](https://www.tensorflow.org/versions/r0.10/how_tos/quantization/index.html).


7. Follow the [TensorFlow iOS Examples Readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples) to run the "simple" sample app. Note that this app uses the Inception v1 model (about 50MB) - it's unfortunate that both TensorFlow for Poets and the TensorFlow [Image Retraining How To](https://www.tensorflow.org/versions/r0.10/how_tos/image_retraining/index.html) posts use the Inception v3 model (about 100MB), which makes running the retrained model on iOS more challenging but necessitates this blog.

8. In the iOS file [RunModelViewController.mm](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ios_examples/simple/RunModelViewController.mm) of the "simple" sample project, make the following 3 changes so you can use the stripped and quantized retrained dog model successfully on an iPhone:
* Add the quantized_stripped_dogs_retrained.pb file generated in step 6 and dog_retrained_labels.txt file generated in step 4 to the project, then replace the two paths:
```
NSString* network_path = FilePathForResourceName(@"tensorflow_inception_graph", @"pb");
...
NSString* labels_path = FilePathForResourceName(@"imagenet_comp_graph_label_strings", @"txt");
```
* Replace the following lines:
```
const int wanted_width = 224;
const int wanted_height = 224;  
const int wanted_channels = 3;  
const float input_mean = 117.0f;  
const float input_std = 1.0f;  
```
with:
```
const int wanted_width = 299;  
const int wanted_height = 299;  
const int wanted_channels = 3;  
const float input_mean = 128.0f;  
const float input_std = 128.0f;  
```
For more info, see this [github issue](https://github.com/tensorflow/tensorflow/issues/2883).
* Replace:
```
std::string input_layer = "input";
std::string output_layer = "output";
```
with:
```
std::string input_layer = "Mul";
std::string output_layer = "final_result";
```
Note the names of input_layer and out_layer are used when running the strip_unused script in step 5.

With these changes, if you use a dog image instead of the default `grace_hopper.jpg` in `NSString* image_path = FilePathForResourceName(@"grace_hopper", @"jpg");` in RunModelViewController.mm, you can run the "simple" app and get the dog breed prediction on your iPhone, and the accuracy would be the same as running the `label_image` script in step 4 on your computer.

To check out my freely available dog breed recognition iPhone app in App Store, using the quantized stripped retrained model, built by following the exact steps above, get it [here](https://itunes.apple.com/us/app/dog-breeds-recognition-powered/id1150923794?mt=8). It's pretty cool to see TensorFlow finally working on my iPhone to solve a real problem, even without Internet connection, after all those steps and months of disillusion, patience, hope, and hard work. Now you can easily replace the dog dataset with another dataset of your interest, such as flowers and plants, to develop your own fun iOS apps.
