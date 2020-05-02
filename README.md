# TFLite-Image
Let's make machine learning simple. <br/>
Building a custom image classifier for your Android application using TensorFlow Lite.

![https://i.imgur.com/lGObtU8.png](https://i.imgur.com/O62Wcof.png)

[ ![Download](https://api.bintray.com/packages/aslam/android/tflite-image/images/download.svg) ](https://bintray.com/aslam/android/tflite-image) [![](https://jitpack.io/v/aslamanver/tflite-image.svg)](https://jitpack.io/#aslamanver/tflite-image) [![Build Status](https://travis-ci.org/aslamanver/tflite-image.svg?branch=master)](https://travis-ci.org/aslamanver/tflite-image)

TFLite-Image for Android - TensorFlow Lite inception model image library for Android

Move your trained model to asset folder or prepare a new image inception model using Google [teachablemachine](https://teachablemachine.withgoogle.com) machine learning library.

You can use the sample inception quant or float model that we used in this project with 299 image dimension.

- [inception_quant.tflite](https://github.com/aslamanver/tflite-image/blob/master/app/src/main/assets/inception_quant.tflite)
- [inception_float.tflite](https://github.com/aslamanver/tflite-image/blob/master/app/src/main/assets/inception_float.tflite)
- [labels.txt](https://github.com/aslamanver/tflite-image/blob/master/app/src/main/assets/labels.txt)<br/>

### Initialization

Add the below dependency into your module level `build.gradle` file

```gradle
implementation 'com.aslam:tflite-image:+'
```

Make sure you have added no compress config for your model files
```gradle
android {
    ....
    aaptOptions {
        noCompress "tflite"
        noCompress "lite"
    }
}
```

### Simple Usage

You need to pass the model file, label text and the model type.

```java
TFLiteImage tfLite = TFLiteImage.getInstance(activity, "your_model_file.tflite", "labels.txt", TFLiteImage.TYPE.QUANT, IMG_DIM_SIZE);
List<Map<String, String>> results = tfLite.predictImage(image view or bitmap image);
```
> `IMG_DIM_SIZE` is 299 or 224 according to your model, you can visualize your model data to check `IMG_DIM_SIZE`.

Inception model types
```java
TFLiteImage.TYPE.QUANT
TFLiteImage.TYPE.FLOAT
```

### Use case

```java
TFLiteImage tfLite = TFLiteImage.getInstance(this, "inception_quant.tflite", "labels.txt", TFLiteImage.TYPE.QUANT);
List<Map<String, String>> results = tfLite.predictImage(binding.imgView);

for (Map<String, String> map : results) {
    Log.e("RESULT", map.get("LABEL") + " - " + map.get("CONFIDENCE"));
}
```

Result

```java
map.get("LABEL");
map.get("CONFIDENCE");
```

Sunglass - 99% <br/>
Glass - 85% <br/>
Jeans - 70% <br/>

### Demonstration
[![Screenshot](/screenshots/1.png)](/screenshots/1.png)

Test our sample app: [TFLite-Image-v1.0.apk](https://drive.google.com/file/d/1YFNNx25bvUhahTkaL_TrRV3MLaQCXedT/view?usp=sharing)

<hr/>

Made with ❤️ by <b>Aslam Anver</b>
