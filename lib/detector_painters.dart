import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:quiver/collection.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.imageSize, this.results, this.liveness);

  final Size? imageSize;
  double scaleX = 0, scaleY = 0; // Initialize scaleX and scaleY
  Multimap <String, Face>? results; // Make results nullable
  late Face face; // Make face non-nullable
  String? liveness = ''; // Make face non-nullable

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;

    if (imageSize == null || results == null) return; // Check for nullability

    for (String label in results!.keys) {
      for (Face face in results![label] as List<Face>) {
        scaleX = size.width / imageSize!.width;
        scaleY = size.height / imageSize!.height;

        final Rect boundingBox = face.boundingBox;
        canvas.drawRRect(
          _scaleRect(
            rect: boundingBox,
            imageSize: imageSize!,
            widgetSize: size,
            scaleX: scaleX,
            scaleY: scaleY,
          ),
          paint,
        );

        TextSpan span = TextSpan(
          style: TextStyle(color: Colors.orange[300], fontSize: 15),
          text: "$label | Liveness: $liveness",
        );
        TextPainter textPainter = TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();

        textPainter.paint(
          canvas,
          Offset(
            size.width - (60 + boundingBox.left.toDouble()) * scaleX,
            (boundingBox.top.toDouble() - 10) * scaleY,
          ),
        );
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

RRect _scaleRect({
  required Rect rect,
  required Size imageSize,
  required Size widgetSize,
  required double scaleX,
  required double scaleY,
}) {
  return RRect.fromLTRBR(
    (widgetSize.width - rect.left.toDouble() * scaleX),
    rect.top.toDouble() * scaleY,
    widgetSize.width - rect.right.toDouble() * scaleX,
    rect.bottom.toDouble() * scaleY,
    Radius.circular(10),
  );
}