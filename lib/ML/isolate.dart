import 'dart:isolate';
import 'package:image/image.dart' as img;
import 'package:realtime_face_recognition/utils/image_converter.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class IsolateInference {
  static const String _debugName = "TFLITE_INFERENCE";
  final ReceivePort _receivePort = ReceivePort();
  late Isolate _isolate;
  late SendPort _sendPort;

  SendPort get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(
      entryPoint,
      _receivePort.sendPort,
      debugName: _debugName,
    );
    _sendPort = await _receivePort.first;
  }

  Future<void> close() async {
    _isolate.kill();
    _receivePort.close();
  }

  static void entryPoint(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);

    await for (final InferenceModel isolateModel in port) {
      var input = imageToArray(isolateModel.image);
      List output = List.filled(1 * 192, 0).reshape([1, 192]);
      final interpreter = Interpreter.fromAddress(
        isolateModel.interpreterAddress,
      );
      interpreter.run(input, output);
      List<double> outputArray = output.first.cast<double>();
      isolateModel.responsePort.send(outputArray);
    }
  }
}

class InferenceModel {
  img.Image image;
  int interpreterAddress;
  SendPort responsePort;

  InferenceModel({
    required this.image,
    required this.interpreterAddress,
    required this.responsePort,
  });
}
