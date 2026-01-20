import 'dart:typed_data';
import 'dart:ui';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class MlKitFaceTestScreen extends StatefulWidget {
  const MlKitFaceTestScreen({super.key});

  @override
  State<MlKitFaceTestScreen> createState() => _MlKitFaceTestScreenState();
}

class _MlKitFaceTestScreenState extends State<MlKitFaceTestScreen> {
  CameraController? _cameraController;
  late FaceDetector _faceDetector;
  bool _isProcessing = false;
  List<Face> _faces = [];

  String _status = "Starting...";

  @override
  void initState() {
    super.initState();
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: true,
        performanceMode: FaceDetectorMode.fast,
      ),
    );
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      debugPrint("📷 Fetching cameras...");
      setState(() => _status = "Fetching cameras");

      final cameras = await availableCameras();
      debugPrint("📷 Cameras found: ${cameras.length}");

      if (cameras.isEmpty) {
        setState(() => _status = "❌ No cameras available");
        return;
      }

      final frontCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      debugPrint("📷 Initializing camera...");
      setState(() => _status = "Initializing camera");

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.low,
        enableAudio: false,
      );

      await _cameraController!.initialize();

      debugPrint("✅ Camera initialized");
      setState(() => _status = "Starting image stream");

      await _cameraController!.startImageStream(_processImage);

      debugPrint("🎥 Image stream started");
      setState(() => _status = "Running");
    } catch (e, st) {
      debugPrint("❌ Camera init failed: $e");
      debugPrintStack(stackTrace: st);
      setState(() => _status = "❌ Error: $e");
    }
  }

  Future<void> _processImage(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final bytes = Uint8List.fromList(
        image.planes.expand((p) => p.bytes).toList(),
      );

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: InputImageRotation.rotation0deg,
          format: InputImageFormat.nv21,
          bytesPerRow: image.planes.first.bytesPerRow,
        ),
      );

      final faces = await _faceDetector.processImage(inputImage);
      setState(() => _faces = faces);
    } catch (e) {
      debugPrint("⚠️ Frame processing error: $e");
    }

    _isProcessing = false;
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("ML Kit Debug Test")),
      body: Stack(
        children: [
          if (_cameraController != null &&
              _cameraController!.value.isInitialized)
            CameraPreview(_cameraController!)
          else
            const Center(child: CircularProgressIndicator()),

          Positioned(
            bottom: 20,
            left: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: Colors.black87,
              child: Text(
                _status,
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),

          CustomPaint(
            painter: FacePainter(_faces),
            child: Container(),
          ),
        ],
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final List<Face> faces;
  FacePainter(this.faces);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    for (final face in faces) {
      final upper = face.contours[FaceContourType.upperLipTop];
      final lower = face.contours[FaceContourType.lowerLipBottom];

      if (upper != null) {
        final path = Path();
        for (int i = 0; i < upper.points.length; i++) {
          final p = upper.points[i];
          if (i == 0) {
            path.moveTo(p.x.toDouble(), p.y.toDouble());
          } else {
            path.lineTo(p.x.toDouble(), p.y.toDouble());
          }
        }
        canvas.drawPath(path, paint);
      }

      if (lower != null) {
        final path = Path();
        for (int i = 0; i < lower.points.length; i++) {
          final p = lower.points[i];
          if (i == 0) {
            path.moveTo(p.x.toDouble(), p.y.toDouble());
          } else {
            path.lineTo(p.x.toDouble(), p.y.toDouble());
          }
        }
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
