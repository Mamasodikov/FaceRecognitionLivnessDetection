import 'dart:math';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

///TODO: Optimize
class FaceAnalyzer {
  final FaceDetector faceDetector;
  Face? previousFace;
  int liveCount = 0;
  int notLiveCount = 0;

  FaceAnalyzer({required this.faceDetector, required this.previousFace});

  void analyzeFace(InputImage inputImage, Function(String result) callback) {
    faceDetector.processImage(inputImage).then((faces) {
      if (faces.length == 1) {

        final currentFace = faces[0];

        if (previousFace != null) {
          final hasMoved = detectMovement(previousFace!, currentFace);
          print("===== Movement detected $hasMoved");
          callback(hasMoved ? "Live" : "Not Live");
        }

      } else {
        callback(
            faces.length > 1 ? "Only one face allowed" : "No face detected");
      }
    }).catchError((error) {
      print("Face Analyzer failed $error");
      callback("Face detection failed");
    });
  }

  bool detectMovement(Face previousFace, Face currentFace) {
    // Reset counts if face changes
    if (previousFace.trackingId != currentFace.trackingId) {
      liveCount = 0;
      notLiveCount = 0;
    }

    // Check if the same face has been detected as live/not live enough times
    if (previousFace.trackingId == currentFace.trackingId && liveCount > 20) {
      return true;
    }
    if (previousFace.trackingId == currentFace.trackingId && notLiveCount > 5) {
      return false;
    }

    // Define a threshold for movement
    const double lowerThreshold = 2;
    const double upperThreshold = 3;

    // Function to calculate the distance between two points
    double distance(Point a, Point b) {
      return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    }

    // Function to check if the distance is within the threshold
    bool isWithinThreshold(double distance) {
      return distance >= lowerThreshold && distance <= upperThreshold;
    }

    // Get landmarks from the previous and current face
    final previousNoseBase =
        previousFace.landmarks[FaceLandmarkType.noseBase]?.position;
    final currentNoseBase =
        currentFace.landmarks[FaceLandmarkType.noseBase]?.position;

    final previousRightEye =
        previousFace.landmarks[FaceLandmarkType.rightEye]?.position;
    final currentRightEye =
        currentFace.landmarks[FaceLandmarkType.rightEye]?.position;

    final previousLeftEye =
        previousFace.landmarks[FaceLandmarkType.leftEye]?.position;
    final currentLeftEye =
        currentFace.landmarks[FaceLandmarkType.leftEye]?.position;

    final previousBottomMouth =
        previousFace.landmarks[FaceLandmarkType.bottomMouth]?.position;
    final currentBottomMouth =
        currentFace.landmarks[FaceLandmarkType.bottomMouth]?.position;

    final previousLeftMouth =
        previousFace.landmarks[FaceLandmarkType.leftMouth]?.position;
    final currentLeftMouth =
        currentFace.landmarks[FaceLandmarkType.leftMouth]?.position;

    final previousRightMouth =
        previousFace.landmarks[FaceLandmarkType.rightMouth]?.position;
    final currentRightMouth =
        currentFace.landmarks[FaceLandmarkType.rightMouth]?.position;

    final previousRightCheek =
        previousFace.landmarks[FaceLandmarkType.rightCheek]?.position;
    final currentRightCheek =
        currentFace.landmarks[FaceLandmarkType.rightCheek]?.position;

    final previousLeftCheek =
        previousFace.landmarks[FaceLandmarkType.leftCheek]?.position;
    final currentLeftCheek =
        currentFace.landmarks[FaceLandmarkType.leftCheek]?.position;

    final previousRightEar =
        previousFace.landmarks[FaceLandmarkType.rightEar]?.position;
    final currentRightEar =
        currentFace.landmarks[FaceLandmarkType.rightEar]?.position;

    final previousLeftEar =
        previousFace.landmarks[FaceLandmarkType.leftEar]?.position;
    final currentLeftEar =
        currentFace.landmarks[FaceLandmarkType.leftEar]?.position;

    // Define the list of landmarks
    final landmarks = [
      {'previous': previousNoseBase, 'current': currentNoseBase},
      {'previous': previousRightEye, 'current': currentRightEye},
      {'previous': previousLeftEye, 'current': currentLeftEye},
      {'previous': previousBottomMouth, 'current': currentBottomMouth},
      {'previous': previousLeftMouth, 'current': currentLeftMouth},
      {'previous': previousRightMouth, 'current': currentRightMouth},
      {'previous': previousRightCheek, 'current': currentRightCheek},
      {'previous': previousLeftCheek, 'current': currentLeftCheek},
      {'previous': previousRightEar, 'current': currentRightEar},
      {'previous': previousLeftEar, 'current': currentLeftEar},
    ];

    // Iterate through the landmarks
    for (final landmark in landmarks) {
      final previous = landmark['previous'];
      final current = landmark['current'];

      // Check if both previous and current landmarks are not null
      if (previous != null && current != null) {
        // Check if the movement is within the threshold
        if (isWithinThreshold(distance(previous, current))) {
          liveCount++;
          return true; // Movement detected
        }
      }
    }

    notLiveCount++;
    return false;
  }
}
