import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

const int WIDTH = 112;
const int HEIGHT = 112;

InputImage getInputImage({
  required CameraImage image,
  required CameraDescription camera,
}) {
  final WriteBuffer allBytes = WriteBuffer();
  for (final Plane plane in image.planes) {
    allBytes.putUint8List(plane.bytes);
  }
  final bytes = allBytes.done().buffer.asUint8List();
  final Size imageSize = Size(
    image.width.toDouble(),
    image.height.toDouble(),
  );
  final imageRotation = InputImageRotationValue.fromRawValue(
    camera.sensorOrientation,
  );
  // if (imageRotation == null) return;

  final inputImageFormat = InputImageFormatValue.fromRawValue(image.format.raw);
  // if (inputImageFormat == null) return null;

  final planeData = image.planes
      .map(
        (plane) => InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        ),
      )
      .toList();

  final inputImageData = InputImageData(
    size: imageSize,
    imageRotation: imageRotation!,
    inputImageFormat: inputImageFormat!,
    planeData: planeData,
  );

  final inputImage = InputImage.fromBytes(
    bytes: bytes,
    inputImageData: inputImageData,
  );

  return inputImage;
}

img.Image convertYUV420ToImage(CameraImage cameraImage) {
  final width = cameraImage.width;
  final height = cameraImage.height;

  final yRowStride = cameraImage.planes[0].bytesPerRow;
  final uvRowStride = cameraImage.planes[0].bytesPerRow;
  final uvPixelStride = cameraImage.planes[0].bytesPerPixel!;

  final image = img.Image(width: width, height: height);

  for (var w = 0; w < width; w++) {
    for (var h = 0; h < height; h++) {
      final uvIndex =
          uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
      final yIndex = h * yRowStride + w;

      final y = cameraImage.planes[0].bytes[yIndex];
      final u = cameraImage.planes[0].bytes[uvIndex];
      final v = cameraImage.planes[0].bytes[uvIndex];

      image.data?.setPixelRgb(w, h, y, y, y);
    }
  }
  return image;
}

int yuv2rgb(int y, int u, int v) {
  // Convert yuv pixel to rgb
  var r = (y + v * 1436 / 1024 - 179).round();
  var g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
  var b = (y + u * 1814 / 1024 - 227).round();

  // Clipping RGB values to be inside boundaries [ 0 , 255 ]
  r = r.clamp(0, 255);
  g = g.clamp(0, 255);
  b = b.clamp(0, 255);

  return 0xff000000 | ((b << 16) & 0xff0000) | ((g << 8) & 0xff00) | (r & 0xff);
}

List<dynamic> imageToArray(img.Image inputImage) {
  img.Image resizedImage =
      img.copyResize(inputImage, width: WIDTH, height: HEIGHT);
  List<double> flattenedList = resizedImage.data!
      .expand((channel) => [channel.r, channel.g, channel.b])
      .map((value) => value.toDouble())
      .toList();
  Float32List float32Array = Float32List.fromList(flattenedList);
  int channels = 3;
  int height = HEIGHT;
  int width = WIDTH;
  Float32List reshapedArray = Float32List(1 * height * width * channels);
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = c * height * width + h * width + w;
        reshapedArray[index] =
            (float32Array[c * height * width + h * width + w] - 127.5) / 127.5;
      }
    }
  }
  return reshapedArray.reshape([1, 112, 112, 3]);
}
