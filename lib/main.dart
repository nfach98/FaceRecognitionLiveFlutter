import 'package:flutter/material.dart';
import 'package:realtime_face_recognition/pages/face_detector_view.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: FaceDetectorView(),
    );
  }
}
