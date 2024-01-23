import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:realtime_face_recognition/ML/recognition.dart';
import 'package:realtime_face_recognition/ML/recognizer.dart';
import 'package:realtime_face_recognition/face_detector_painter.dart';
import 'package:realtime_face_recognition/utils/image_converter.dart';

class FaceDetectorView extends StatefulWidget {
  const FaceDetectorView({super.key});

  @override
  State<FaceDetectorView> createState() => _FaceDetectorViewState();
}

class _FaceDetectorViewState extends State<FaceDetectorView> {
  CameraController? _controller;
  late FaceDetector faceDetector;
  late Recognizer recognizer;
  bool _isBusy = false;
  bool _isRegister = false;

  final _nameController = TextEditingController();
  final recs = <Recognition>[];

  @override
  void initState() {
    SystemChrome.setEnabledSystemUIMode(
      SystemUiMode.manual,
      overlays: [SystemUiOverlay.top],
    );
    super.initState();
    faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.fast,
        enableContours: true,
        enableLandmarks: true,
      ),
    );
    recognizer = Recognizer(numThreads: 4);
    _initializeCamera();
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  _initializeCamera() async {
    final cameras = await availableCameras();
    _controller = CameraController(
      cameras.where((e) => e.lensDirection == CameraLensDirection.front).first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.nv21
          : ImageFormatGroup.bgra8888,
    );
    _controller?.addListener(() {
      if (_isBusy) {
        _controller?.stopImageStream();
      } else {
        _controller?.startImageStream(_processCameraImage);
      }
    });
    await _controller?.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _controller?.startImageStream(_processCameraImage);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(body: _buildCamera());
  }

  Widget _buildCamera() {
    if (_controller == null) return Container();
    if (_controller?.value.isInitialized == false) return Container();

    final size = MediaQuery.of(context).size;
    var scale = size.aspectRatio * (_controller?.value.aspectRatio ?? 1);
    if (scale < 1) scale = 1 / scale;

    return Transform.scale(
      scale: scale,
      child: OverflowBox(
        alignment: Alignment.center,
        child: FittedBox(
          fit: BoxFit.fitHeight,
          child: SizedBox(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.width *
                _controller!.value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_controller!),
                if (recs.isNotEmpty)
                  Positioned.fill(
                    child: SizedBox(
                      width: MediaQuery.of(context).size.width,
                      height: MediaQuery.of(context).size.width *
                          _controller!.value.aspectRatio,
                      child: _buildResult(recs),
                    ),
                  ),
                if (recs.where((e) => _isUnknown(e.distance)).isNotEmpty)
                  Positioned(
                    bottom: 100,
                    height: 80,
                    left: 0,
                    right: 0,
                    child: _buildBottom(),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isBusy) return;
    _isBusy = true;
    if (_controller != null) {
      final inputImage = getInputImage(
        image: image,
        camera: _controller!.description,
      );
      final faces = await faceDetector.processImage(inputImage);
      final runStart = DateTime.now().millisecondsSinceEpoch;
      _performFaceRecognition(faces, image);
      final runEnd = DateTime.now().millisecondsSinceEpoch - runStart;
      print('Time to run inference: $runEnd ms');
    }
  }

  Future<void> _performFaceRecognition(
    List<Face> faces,
    CameraImage frame,
  ) async {
    final image = convertYUV420ToImage(frame);
    final imgRotated = img.copyRotate(image, angle: 270);

    recs.clear();
    for (var i = 0; i < faces.length; i++) {
      final face = faces[i];
      final faceRect = face.boundingBox;
      img.Image croppedFace = img.copyCrop(
        imgRotated,
        x: faceRect.left.toInt(),
        y: faceRect.top.toInt(),
        width: faceRect.width.toInt(),
        height: faceRect.height.toInt(),
      );
      final recognition = await recognizer.recognize(
        croppedFace,
        face.boundingBox,
      );
      recs.add(recognition);
      if (_isUnknown(recognition.distance)) recognition.name = '';
      if (_isRegister) {
        _showFaceRegistrationDialogue(
          image: croppedFace,
          recognition: recognition,
        );
        _isRegister = false;
      }
    }

    setState(() {
      _isBusy = false;
    });
  }

  Widget? _buildResult(List<Recognition> rec) {
    if (_controller == null || _controller?.value.isInitialized == false) {
      return null;
    }
    final Size imageSize = Size(
      _controller!.value.previewSize!.height,
      _controller!.value.previewSize!.width,
    );
    CustomPainter painter = FaceDetectorPainter(
      imageSize,
      rec,
      _controller!.description.lensDirection,
    );
    return CustomPaint(painter: painter);
  }

  Widget _buildBottom() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 20),
      decoration: const BoxDecoration(
        color: Colors.blue,
        shape: BoxShape.circle,
      ),
      child: IconButton(
        icon: const Icon(
          Icons.face_retouching_natural,
          color: Colors.white,
        ),
        iconSize: 40,
        color: Colors.black,
        onPressed: () => _isRegister = true,
      ),
    );
  }

  _showFaceRegistrationDialogue({
    required img.Image image,
    required Recognition recognition,
  }) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Face Registration", textAlign: TextAlign.center),
        alignment: Alignment.center,
        content: SizedBox(
          height: 340,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(
                height: 20,
              ),
              Image.memory(
                Uint8List.fromList(img.encodeBmp(image)),
                width: 200,
                height: 200,
              ),
              SizedBox(
                width: 200,
                child: TextField(
                  controller: _nameController,
                  decoration: const InputDecoration(
                    fillColor: Colors.white,
                    filled: true,
                    hintText: 'Enter Name',
                  ),
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                onPressed: () {
                  recognizer.registerFaceInDB(
                    _nameController.text,
                    recognition.embeddings,
                  );
                  _nameController.clear();
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text("Face Registered"),
                  ));
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  minimumSize: const Size(200, 40),
                ),
                child: const Text("Register"),
              ),
            ],
          ),
        ),
        contentPadding: EdgeInsets.zero,
      ),
    );
  }

  bool _isUnknown(double value) => value >= .15;
}
