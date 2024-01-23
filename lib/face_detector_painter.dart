import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:realtime_face_recognition/ML/recognition.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.absoluteImageSize, this.faces, this.camDire2);

  final Size absoluteImageSize;
  final List<Recognition> faces;
  CameraLensDirection camDire2;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;
    const strokeWidth = 8.0;

    for (Recognition face in faces) {
      final color = face.name.isEmpty ? Colors.red : Colors.indigoAccent;

      final Paint paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..color = color;

      final l = camDire2 == CameraLensDirection.front
          ? (absoluteImageSize.width - face.location.right) * scaleX +
              strokeWidth
          : face.location.left * scaleX;
      final r = camDire2 == CameraLensDirection.front
          ? (absoluteImageSize.width - face.location.left) * scaleX -
              strokeWidth
          : face.location.right * scaleX;

      canvas.drawRRect(
        RRect.fromLTRBR(
          l,
          face.location.top * scaleY,
          r,
          face.location.bottom * scaleY,
          const Radius.circular(20),
        ),
        paint,
      );
      canvas.drawRRect(
        RRect.fromLTRBR(
          l,
          face.location.top * scaleY,
          r,
          face.location.top * scaleY + 32,
          const Radius.circular(20),
        ),
        paint..style = PaintingStyle.fill,
      );
      canvas.drawRect(
        Rect.fromLTRB(
          l,
          face.location.top * scaleY + 20,
          r,
          face.location.top * scaleY + 32,
        ),
        paint..style = PaintingStyle.fill,
      );

      TextSpan span = TextSpan(
        style: TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w600,
          backgroundColor: color,
        ),
        text: '${face.name} ${((1 - face.distance) * 100).toStringAsFixed(2)}%',
      );
      TextPainter tp = TextPainter(
        text: span,
        textAlign: TextAlign.center,
        textDirection: TextDirection.ltr,
      );

      tp.layout();
      tp.paint(
        canvas,
        Offset(
          (l + face.location.width * scaleX / 2) -
              tp.width * scaleX +
              strokeWidth * 2,
          face.location.top * scaleY + 4,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return true;
  }
}
