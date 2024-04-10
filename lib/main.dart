import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:ml_kit_face/generated/assets.dart';
import 'package:path_provider/path_provider.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'detector_painters.dart';
import 'face_analyzer.dart';
import 'utils.dart';
import 'package:image/image.dart' as imglib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:quiver/collection.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(MaterialApp(
    themeMode: ThemeMode.light,
    theme: ThemeData(brightness: Brightness.light),
    home: _MyHomePage(),
    title: "Face Recognition",
    debugShowCheckedModeBanner: false,
  ));
}

class _MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<_MyHomePage> {
  late File jsonFile;
  Multimap<String, Face>? _scanResults;
  late CameraController _camera;
  late var interpreter;
  bool _isDetecting = false;
  CameraLensDirection _direction = CameraLensDirection.front;
  Map<String, dynamic> data = {};
  double threshold = 1.0;
  late Directory tempDir;
  List<double> e1 = [];
  bool _faceFound = false;
  final TextEditingController _name = TextEditingController();
  late Future<void> cameraInit;
  Face? previousFace;
  String liveness = 'Waiting...';
  int liveCount = 0;
  int notLiveCount = 0;

  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    cameraInit = _initializeCamera();
  }

  Future<void> loadModel() async {
    try {
      final gpuDelegateV2 =
          tfl.GpuDelegateV2(options: tfl.GpuDelegateOptionsV2());

      var interpreterOptions = tfl.InterpreterOptions()
        ..addDelegate(gpuDelegateV2);
      interpreter = await tfl.Interpreter.fromAsset(Assets.assetsMobilefacenet,
          options: interpreterOptions);
    } on Exception {
      print('Failed to load model.');
    }
  }

  Future<void> _initializeCamera() async {
    await loadModel();
    CameraDescription description = await getCamera(_direction);

    InputImageRotation rotation = rotationIntToImageRotation(
      description.sensorOrientation,
    );

    _camera =
        CameraController(description, ResolutionPreset.low, enableAudio: false);
    await _camera.initialize();
    await Future.delayed(const Duration(milliseconds: 500));
    tempDir = await getApplicationDocumentsDirectory();
    String _embPath = '${tempDir.path}/emb.json';
    jsonFile = File(_embPath);
    if (jsonFile.existsSync()) data = json.decode(jsonFile.readAsStringSync());

    _camera.startImageStream((CameraImage image) {
      if (_isDetecting) return;
      _isDetecting = true;
      String res;
      final finalResult = Multimap<String, Face>();
      final FaceDetector faceDetector = FaceDetector(
          options: FaceDetectorOptions(
              performanceMode: FaceDetectorMode.accurate,
              enableLandmarks: true));

      var inputImage = InputImage.fromBytes(
        bytes: Uint8List.fromList(
          image.planes.fold(
              <int>[],
              (List<int> previousValue, element) =>
                  previousValue..addAll(element.bytes)),
        ),
        metadata: buildMetaData(image, rotation),
      );

      // FaceAnalyzer(faceDetector: faceDetector, previousFace: previousFace)
      //     .analyzeFace(inputImage, (result) {
      //   liveness = result;
      // });

      analyzeFace(inputImage, (result) {
        liveness = result;
      }, faceDetector);

      detect(image, _getDetectionMethod(faceDetector), rotation).then(
        (result) async {
          if (result.length == 0) {
            _faceFound = false;
          } else {
            _faceFound = true;
          }

          imglib.Image convertedImage = _convertCameraImage(image, _direction);
          for (Face face in result) {
            double x, y, w, h;
            x = (face.boundingBox.left - 10);
            y = (face.boundingBox.top - 10);
            w = (face.boundingBox.width + 10);
            h = (face.boundingBox.height + 10);
            imglib.Image croppedImage = imglib.copyCrop(
                convertedImage, x.round(), y.round(), w.round(), h.round());
            croppedImage = imglib.copyResizeCropSquare(croppedImage, 112);
            // int startTime = new DateTime.now().millisecondsSinceEpoch;
            res = _recognize(croppedImage);
            // int endTime = new DateTime.now().millisecondsSinceEpoch;
            // print("Inference took ${endTime - startTime}ms");
            finalResult.add(res, face);
          }
          setState(() {
            _scanResults = finalResult;
          });

          _isDetecting = false;
        },
      ).catchError(
        (_) {
          _isDetecting = false;
        },
      );
    });
  }

  ///TODO: Optimize (Fix)
  void analyzeFace(InputImage inputImage, Function(String result) callback,
      FaceDetector faceDetector) {
    faceDetector.processImage(inputImage).then((faces) {
      if (faces.length == 1) {
        final currentFace = faces[0];

        if (previousFace != null) {
          final hasMoved = detectMovement(previousFace!, currentFace);
          print("===== Movement detected $hasMoved");
          callback(hasMoved ? "Live" : "Not Live");
        }
        previousFace = currentFace;
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
    if (previousFace.trackingId == currentFace.trackingId && liveCount > 10) {
      liveCount = 0;
      notLiveCount = 0;
      return true;
    }
    if (previousFace.trackingId == currentFace.trackingId && notLiveCount > 5) {
      liveCount = 0;
      notLiveCount = 0;
      return false;
    }

    // Define a threshold for movement
    const double lowerThreshold = 2;
    const double upperThreshold = 2.5;

    // Function to calculate the distance between two points
    // double distance(Point a, Point b) {
    //   return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    // }

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

    // landmarks.forEach((landmark) {
    //   final previousPosition = landmark['previous'];
    //   final currentPosition = landmark['current'];
    //
    //   final deltaX = (previousPosition?.x ?? 0 - currentPosition!.x).abs();
    //   final deltaY = (previousPosition?.y ?? 0 - currentPosition.y).abs();
    //
    //   if (isWithinThreshold(deltaX) || isWithinThreshold(deltaY)) {
    //     liveCount++;
    //     return true;
    //   }

    ///Iterate through the landmarks
    for (final landmark in landmarks) {
      final previous = landmark['previous'];
      final current = landmark['current'];

      final deltaX = (previous?.x ?? 0 - current!.x).abs();
      final deltaY = (previous?.y ?? 0 - current!.y).abs();

      print('========== DELTAX: ${deltaX / 100}');
      print('========== DELTAY: ${deltaY / 100}');

      // Check if both previous and current landmarks are not null
      if (previous != null && current != null) {
        // Check if the movement is within the threshold
        if (isWithinThreshold(deltaX / 100) ||
            isWithinThreshold(deltaY / 100)) {
          liveCount++;
          return true;
        }
      }
    }

    notLiveCount++;
    return false;
  }

  HandleDetection _getDetectionMethod(FaceDetector faceDetector) {
    return faceDetector.processImage;
  }

  Widget _buildResults() {
    const Text noResultsText = Text('');
    if (_scanResults == null || !_camera.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      _camera.value.previewSize?.height ?? 0,
      _camera.value.previewSize?.width ?? 0,
    );
    painter = FaceDetectorPainter(imageSize, _scanResults!, liveness);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    return FutureBuilder<void>(
      future: cameraInit,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(
            child: CircularProgressIndicator(),
          );
        } else if (snapshot.connectionState == ConnectionState.done) {
          if (!_camera.value.isInitialized) {
            return const Center(
              child: Text('Waiting for the camera...'),
            );
          }
          return Container(
            constraints: const BoxConstraints.expand(),
            child: Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(_camera),
                _buildResults(),
              ],
            ),
          );
        } else {
          return const Center(
            child: Text('Something went wrong.'),
          );
        }
      },
    );
  }

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
    } else {
      _direction = CameraLensDirection.back;
    }
    await _camera.stopImageStream();
    // await _camera.dispose();

    _initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face recognition'),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete) {
                _resetFile();
              } else {
                _viewLabels();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                value: Choice.view,
                child: Text('View Saved Faces'),
              ),
              const PopupMenuItem<Choice>(
                value: Choice.delete,
                child: Text('Remove all faces'),
              )
            ],
          ),
        ],
      ),
      body: _buildImage(),
      floatingActionButton:
          Column(mainAxisAlignment: MainAxisAlignment.end, children: [
        FloatingActionButton(
          backgroundColor: (_faceFound) ? Colors.blue : Colors.blueGrey,
          onPressed: () {
            if (_faceFound) _addLabel();
          },
          heroTag: null,
          child: const Icon(Icons.add),
        ),
        const SizedBox(
          height: 10,
        ),
        FloatingActionButton(
          onPressed: _toggleCameraDirection,
          heroTag: null,
          child: _direction == CameraLensDirection.back
              ? const Icon(Icons.add_reaction)
              : const Icon(Icons.camera),
        ),
      ]),
    );
  }

  imglib.Image _convertCameraImage(CameraImage image, CameraLensDirection dir) {
    int width = image.width;
    int height = image.height;
    // imglib -> Image package from https://pub.dartlang.org/packages/image
    var img = imglib.Image(width, height); // Create Image buffer
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 0;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (dir == CameraLensDirection.front)
        ? imglib.copyRotate(img, -90)
        : imglib.copyRotate(img, 90);
    return img1;
  }

  String _recognize(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, 0).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    return compare(e1).toUpperCase();
  }

  String compare(List currEmb) {
    if (data.isEmpty) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    print("$minDist $predRes");
    return predRes;
  }

  void _resetFile() {
    data = {};
    jsonFile.deleteSync();
  }

  void _viewLabels() {
    String name;
    var alert = AlertDialog(
      title: const Text("Saved Faces"),
      content: ListView.builder(
          padding: const EdgeInsets.all(2),
          itemCount: data.length,
          itemBuilder: (BuildContext context, int index) {
            name = data.keys.elementAt(index);
            return Column(
              children: <Widget>[
                ListTile(
                  title: Text(
                    name,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey[400],
                    ),
                  ),
                ),
                const Padding(
                  padding: EdgeInsets.all(2),
                ),
                const Divider(),
              ],
            );
          }),
      actions: <Widget>[
        ElevatedButton(
          child: const Text("OK"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _addLabel() {
    print("Adding new face");
    var alert = AlertDialog(
      title: const Text("Add Face"),
      content: Row(
        children: <Widget>[
          Expanded(
            child: TextField(
              controller: _name,
              autofocus: true,
              decoration: const InputDecoration(
                  labelText: "Name", icon: Icon(Icons.face)),
            ),
          )
        ],
      ),
      actions: <Widget>[
        ElevatedButton(
            child: const Text("Save"),
            onPressed: () {
              _handle(_name.text.toUpperCase());
              _name.clear();
              Navigator.pop(context);
            }),
        ElevatedButton(
          child: const Text("Cancel"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _handle(String text) {
    data[text] = e1;
    jsonFile.writeAsStringSync(json.encode(data));
    _initializeCamera();
  }
}
