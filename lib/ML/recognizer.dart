import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui';
import 'package:document_analysis/document_analysis.dart';
import 'package:image/image.dart' as img;
import 'package:realtime_face_recognition/ML/isolate.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../DB/DatabaseHelper.dart';
import 'recognition.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  static const int WIDTH = 112;
  static const int HEIGHT = 112;
  final dbHelper = DatabaseHelper();
  final isolateInference = IsolateInference();
  Map<String, Recognition> registered = {};
  final modelName = 'assets/mobile_face_net.tflite';

  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (Platform.isAndroid) {
      _interpreterOptions.addDelegate(GpuDelegateV2());
      _interpreterOptions.useNnApiForAndroid = true;
    }
    if (Platform.isIOS) {
      _interpreterOptions.addDelegate(GpuDelegate());
    }
    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    loadModel();
    initDB();
  }

  initDB() async {
    await dbHelper.init();
    loadRegisteredFaces();
    startIsolate();
  }

  startIsolate() async {
    await isolateInference.start();
  }

  void loadRegisteredFaces() async {
    registered.clear();
    final allRows = await dbHelper.queryAllRows();
    // debugPrint('query all rows:');
    for (final row in allRows) {
      //  debugPrint(row.toString());
      print(row[DatabaseHelper.columnName]);
      String name = row[DatabaseHelper.columnName];
      List<double> embd = row[DatabaseHelper.columnEmbedding]
          .split(',')
          .map((e) => double.parse(e))
          .toList()
          .cast<double>();
      Recognition recognition =
          Recognition(row[DatabaseHelper.columnName], Rect.zero, embd, 0);
      registered.putIfAbsent(name, () => recognition);
      print("R=$name");
    }
  }

  void registerFaceInDB(String name, List<double> embedding) async {
    // row to insert
    Map<String, dynamic> row = {
      DatabaseHelper.columnName: name,
      DatabaseHelper.columnEmbedding: embedding.join(',')
    };
    final id = await dbHelper.insert(row);
    print('inserted row id: $id');
    loadRegisteredFaces();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(modelName);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
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
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, WIDTH, HEIGHT, 3]);
  }

  Future<Recognition> recognize(img.Image image, Rect location) async {
    ReceivePort responsePort = ReceivePort();
    isolateInference.sendPort.send(InferenceModel(
      image: image,
      interpreterAddress: interpreter.address,
      responsePort: responsePort.sendPort,
    ));
    var output = await responsePort.first;

    // var input = imageToArray(image);
    // List output = List.filled(1 * 192, 0).reshape([1, 192]);
    // interpreter.run(input, output);

    Pair pair = findNearest(output);
    print("distance= ${pair.distance}");

    return Recognition(pair.name, location, output, pair.distance);
  }

  findNearest(List<double> emb) {
    Pair pair = Pair('', 1);
    for (MapEntry<String, Recognition> item in registered.entries) {
      final name = item.key;
      final knownEmb = item.value.embeddings;
      var distance = 0.0;

      /// Cosine distance
      distance = cosineDistance(knownEmb, emb);
      if (pair.distance == 1 || distance.abs() < pair.distance) {
        pair.distance = distance.abs();
        pair.name = name;
      }

      /// Euclidean distance
      // for (int i = 0; i < emb.length; i++) {
      //   double diff = emb[i] - knownEmb[i];
      //   distance += diff * diff;
      // }
      // distance = sqrt(distance);
    }
    return pair;
  }

  void close() {
    interpreter.close();
  }
}

class Pair {
  String name;
  double distance;
  Pair(this.name, this.distance);
}
