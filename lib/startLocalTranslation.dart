import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vosk_flutter/vosk_flutter.dart'; 
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:camera/camera.dart';
import 'package:socket_io_client/socket_io_client.dart' as IO;
import 'package:record/record.dart';
import 'package:image/image.dart' as img;

class LocalTranscriptionScreen extends StatefulWidget {
  const LocalTranscriptionScreen({super.key});
  @override
  State<LocalTranscriptionScreen> createState() => _LocalTranscriptionScreenState();
}

class _LocalTranscriptionScreenState extends State<LocalTranscriptionScreen> {
  final VoskFlutterPlugin _vosk = VoskFlutterPlugin.instance();
  Model? _model;
  Recognizer? _recognizer;
  SpeechService? _speechService;

  // Camera for OBS Virtual Camera streaming
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  int _selectedCameraIndex = -1;

  // WebSocket for real-time communication
  IO.Socket? _socket;
  bool _isServerConnected = false;

  // Audio streaming
  AudioRecorder? _audioRecorder;
  StreamSubscription? _audioStreamSubscription;
  List<int> _audioBuffer = [];

  // Frame streaming stats
  int _frameCounter = 0;
  int _audioChunksSent = 0;
  DateTime? _streamStartTime;

  // Lip preview from server
  Uint8List? _lipPreviewImage;
  int _lipFramesReceived = 0;

  // Server configuration
  String _serverIp = '192.168.254.104';
  final int _serverPort = 5000;

  // Controllers
  final ScrollController _scrollController = ScrollController();

  // State
  String _status = 'Initializing Model...';
  String _transcribedText = '';
  String _interimText = '';
  String _simulatedLipText = '';
  bool _isRecognizing = false;
  bool _modelLoaded = false;

  // Audio silence detection for simulated lip reading
  DateTime? _lastAudioDetectedTime;
  Timer? _audioSilenceTimer;  // NEW: Timer for tracking audio silence
  
  // NEW: Server-side lip reading prediction tracking
  bool _hasAudioInput = false;  // Track if server detects audio
  bool _hasLipDetection = false;  // Track if server detects lips in video
  String _serverLipPrediction = '';  // Full lip reading text from server
  String _latestLipWord = '';  // Latest word from server
  
  // Audio-Visual mode (both audio and video)
  String _avModeText = '';  // Text for AV mode with simulated delay
  Timer? _avDelayTimer;  // Timer for simulating video processing delay

  // Display Settings
  double _displayTextSize = 16.0;
  Color _displayTextColor = Colors.black;
  TextAlign _displayTextAlignment = TextAlign.center;
  Alignment _displayContainerAlignment = Alignment.center;

  // Model path
  final String _modelAssetPath = 'assets/models/smartsense_model.zip';

  // Frame rate control
  bool _isProcessingFrame = false;
  DateTime? _lastFrameSentTime;  // NEW: Track when last frame was sent
  final int _targetFrameRate = 30;  // NEW: Target 30fps to match server

  @override
  void initState() {
    super.initState();
    _audioRecorder = AudioRecorder();
    _loadDisplaySettings();
    _loadServerConfig().then((_) {
      _connectToServer();
      _initializeCamera();
    });
    _initVosk();
  }

  Future<void> _loadDisplaySettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _displayTextSize = prefs.getDouble('textSize') ?? 16.0;
      int? colorVal = prefs.getInt('textColorValue');
      _displayTextColor = colorVal != null ? Color(colorVal) : Colors.black;

      String alignString = prefs.getString('textAlignment') ?? 'center';
      if (alignString.contains('Start')) {
        _displayTextAlignment = TextAlign.left;
      } else if (alignString.contains('End')) {
        _displayTextAlignment = TextAlign.right;
      } else {
        _displayTextAlignment = TextAlign.center;
      }

      switch (alignString) {
        case 'topStart': _displayContainerAlignment = Alignment.topLeft; break;
        case 'topCenter': _displayContainerAlignment = Alignment.topCenter; break;
        case 'topEnd': _displayContainerAlignment = Alignment.topRight; break;
        case 'centerStart': _displayContainerAlignment = Alignment.centerLeft; break;
        case 'center': _displayContainerAlignment = Alignment.center; break;
        case 'centerEnd': _displayContainerAlignment = Alignment.centerRight; break;
        case 'bottomStart': _displayContainerAlignment = Alignment.bottomLeft; break;
        case 'bottomCenter': _displayContainerAlignment = Alignment.bottomCenter; break;
        case 'bottomEnd': _displayContainerAlignment = Alignment.bottomRight; break;
        default: _displayContainerAlignment = Alignment.center;
      }
    });
  }

  Future<void> _loadServerConfig() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _serverIp = prefs.getString('serverIp') ?? '192.168.254.104';
      _selectedCameraIndex = prefs.getInt('selectedCamera') ?? -1;
    });
  }

  Future<void> _saveServerConfig() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('serverIp', _serverIp);
    await prefs.setInt('selectedCamera', _selectedCameraIndex);
  }

  void _connectToServer() {
    if (_socket != null) {
      _socket!.disconnect();
      _socket!.dispose();
    }

    try {
      debugPrint('🔌 Connecting to ws://$_serverIp:$_serverPort');
      
      _socket = IO.io('http://$_serverIp:$_serverPort', 
        IO.OptionBuilder()
          .setTransports(['websocket'])
          .enableAutoConnect()
          .enableReconnection()
          .setReconnectionDelay(1000)
          .build()
      );

      _socket!.onConnect((_) {
        debugPrint('✅ WebSocket connected!');
        if (mounted) {
          setState(() {
            _isServerConnected = true;
            _status = 'AV Model Connected';
          });
        }
      });

      _socket!.onDisconnect((_) {
        debugPrint('❌ WebSocket disconnected');
        if (mounted) {
          setState(() {
            _isServerConnected = false;
            _status = 'Disconnected - No Server';
          });
        }
      });

      _socket!.on('lip_preview', (data) {
        _handleLipPreview(data);
      });

      // NEW: Listen for real-time predictions from server
      _socket!.on('chunk_processed', (data) {
        _handleServerPrediction(data);
      });

      _socket!.on('frame_processed', (data) {
        // Optional: handle frame processing confirmation
        debugPrint('📸 Frame processed: ${data['frames_buffered']} buffered');
      });

      _socket!.connect();
      
    } catch (e) {
      debugPrint('❌ WebSocket error: $e');
    }
  }

  void _disconnectFromServer() {
    if (_socket != null) {
      _socket!.disconnect();
      _socket!.dispose();
      _socket = null;
    }
    
    if (mounted) {
      setState(() {
        _isServerConnected = false;
        _status = 'Disconnected - No Server';
        _lipPreviewImage = null;
      });
    }
    
    debugPrint('🔌 Disconnected from server');
  }

  void _handleLipPreview(dynamic data) {
    try {
      final lipImageBase64 = data['lip_image'];
      if (lipImageBase64 != null && lipImageBase64 is String) {
        final bytes = base64Decode(lipImageBase64);
        if (mounted) {
          setState(() {
            _lipPreviewImage = bytes;
            _lipFramesReceived++;
          });
        }
      }
    } catch (e) {
      debugPrint('❌ Error handling lip preview: $e');
    }
  }

  // NEW: Handle real-time predictions from server
  void _handleServerPrediction(dynamic data) {
    try {
      final prediction = data['prediction'];
      if (prediction != null && prediction is Map) {
        final text = prediction['text'] ?? '';
        final newWord = prediction['new_word'];
        final hasAudio = prediction['has_audio'] ?? false;
        final hasLips = prediction['has_lips'] ?? false;  // NEW: Get lips detection from server
        final source = prediction['source'] ?? 'none';
        
        if (mounted) {
          setState(() {
            _hasAudioInput = hasAudio;
            _hasLipDetection = hasLips;  // NEW: Update lip detection status from server
            
            // Update full lip prediction text
            if (text.isNotEmpty) {
              _serverLipPrediction = text;
            }
            
            // If a new word was detected from lip movement
            if (newWord != null && newWord is String && newWord.isNotEmpty) {
              _latestLipWord = newWord;
              
              // Add to transcript ONLY if no audio is present (pure lip reading mode)
              if (!hasAudio && source == 'visual' && hasLips) {
                _transcribedText += '$newWord ';
                _scrollToBottom();
                
                debugPrint('💬 Server Lip Prediction: "$newWord" (source: $source, lips: $hasLips)');
              }
              
              // Clear the latest word after 1.5 seconds
              Future.delayed(const Duration(milliseconds: 1500), () {
                if (mounted && _latestLipWord == newWord) {
                  setState(() {
                    _latestLipWord = '';
                  });
                }
              });
            }
          });
        }
      }
      
      // Handle audio features
      final audioFeatures = data['audio_features'];
      if (audioFeatures != null && audioFeatures is Map) {
        final hasAudio = audioFeatures['has_audio'] ?? false;
        if (mounted) {
          setState(() {
            _hasAudioInput = hasAudio;
          });
        }
      }
      
    } catch (e) {
      debugPrint('❌ Error handling server prediction: $e');
    }
  }

  Future<void> _initializeCamera() async {
    var cameraStatus = await Permission.camera.request();
    if (!cameraStatus.isGranted) {
      debugPrint('Camera permission denied');
      return;
    }

    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        debugPrint('No camera found');
        return;
      }

      debugPrint('📷 Available cameras:');
      for (int i = 0; i < _cameras!.length; i++) {
        debugPrint('  [$i] ${_cameras![i].name} - ${_cameras![i].lensDirection}');
      }

      CameraDescription selectedCamera;
      if (_selectedCameraIndex >= 0 && _selectedCameraIndex < _cameras!.length) {
        selectedCamera = _cameras![_selectedCameraIndex];
        debugPrint('✅ Using saved camera: ${selectedCamera.name}');
      } else {
        selectedCamera = _cameras!.firstWhere(
          (camera) => camera.name.toLowerCase().contains('obs') || 
                      camera.name.toLowerCase().contains('virtual'),
          orElse: () => _cameras!.first,
        );
        debugPrint('✅ Auto-selected camera: ${selectedCamera.name}');
      }

      _cameraController = CameraController(
        selectedCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();
      
      if (mounted) {
        setState(() => _isCameraInitialized = true);
      }
      debugPrint('✅ Camera ready: ${selectedCamera.name}');
    } catch (e) {
      debugPrint('❌ Camera error: $e');
      if (mounted) {
        setState(() => _isCameraInitialized = false);
      }
    }
  }

  Future<void> _initVosk() async {
    var status = await Permission.microphone.request();
    if (!status.isGranted) {
      setState(() => _status = 'Mic Permission Denied');
      return;
    }

    try {
      setState(() => _status = 'Verifying Asset Bundle...');

      try {
        final byteData = await rootBundle.load(_modelAssetPath);
        debugPrint("✅ Asset found: $_modelAssetPath");
        debugPrint("📦 Size: ${(byteData.lengthInBytes / 1024 / 1024).toStringAsFixed(2)} MB");
        
        if (byteData.lengthInBytes < 1000) {
           throw Exception("File too small. Check the zip file.");
        }
      } catch (e) {
        debugPrint("Asset Load Error: $e");
        throw Exception("ASSET NOT FOUND. Check pubspec.yaml.");
      }

      setState(() => _status = 'Extracting Model...');
      
      String modelPath = await ModelLoader().loadFromAssets(_modelAssetPath);
      
      final dir = Directory(modelPath);
      
      if (!await dir.exists()) {
        final parentDir = dir.parent;
        if (await parentDir.exists()) {
           final children = parentDir.listSync();
           bool hasConf = children.any((e) => e.path.endsWith("/conf") || e.path.endsWith("\\conf") || e.path.endsWith("conf"));
           
           if (hasConf) {
             modelPath = parentDir.path;
           } else {
             throw FileSystemException("'conf' folder not found.", modelPath);
           }
        }
      } else {
        final List<FileSystemEntity> children = dir.listSync();
        if (children.length == 1 && children.first is Directory) {
          modelPath = children.first.path;
        }
      }

      _model = await _vosk.createModel(modelPath);
      _recognizer = await _vosk.createRecognizer(model: _model!, sampleRate: 16000);

      if (_recognizer != null) {
        try {
          await _speechService?.dispose();
        } catch (e) {
          debugPrint("Cleaning up: $e");
        }
        
        _speechService = await _vosk.initSpeechService(_recognizer!);
        _setupListeners();
        setState(() {
          _modelLoaded = true;
          _status = _isServerConnected ? 'Ready (AV Model)' : 'Ready (Connect Server)';
        });
      }
    } catch (e) {
      debugPrint("❌ Error: $e");
      setState(() => _status = 'Load Failed: ${e.toString().substring(0, 50)}...');
    }
  }

  void _setupListeners() {
    _speechService!.onPartial().listen((partialJson) {
      Map<String, dynamic> data = jsonDecode(partialJson);
      final partial = data['partial'] ?? '';
      
      if (mounted) {
        setState(() {
          // Update audio input status when Vosk detects speech
          if (partial.isNotEmpty) {
            // NEW: Clear stale lip predictions when audio starts
            if (!_hasAudioInput) {
              _latestLipWord = '';
              _serverLipPrediction = '';
              debugPrint('🔊 Audio started - Clearing stale lip predictions');
            }
            
            _hasAudioInput = true;
            _lastAudioDetectedTime = DateTime.now();
            
            // NEW: If both audio and lips detected, use Audio-Visual mode with delay
            if (_hasLipDetection) {
              // Store partial for delayed AV mode display (300ms delay to simulate video processing)
              _avDelayTimer?.cancel();
              _avDelayTimer = Timer(const Duration(milliseconds: 300), () {
                if (mounted) {
                  setState(() {
                    _avModeText = partial;
                  });
                }
              });
            } else {
              // Pure audio mode - show immediately
              _interimText = partial;
            }
          }
        });
      }
    });

    _speechService!.onResult().listen((resultJson) {
      Map<String, dynamic> data = jsonDecode(resultJson);
      String text = data['text'] ?? '';
      
      if (text.isNotEmpty && mounted) {
        setState(() {
          _hasAudioInput = true;
          _lastAudioDetectedTime = DateTime.now();
          
          
          if (_hasLipDetection) {
            _transcribedText += ' $text\n';
          } else {
            _transcribedText += ' $text\n';
          }
          
          _interimText = '';
          _avModeText = '';
          _scrollToBottom();
        });
      }
    });
    
    // Add a timer to reset audio input flag after 2 seconds of silence
    _audioSilenceTimer?.cancel();
    _audioSilenceTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (!_isRecognizing) {
        timer.cancel();
        return;
      }
      
      if (_lastAudioDetectedTime != null) {
        final silenceDuration = DateTime.now().difference(_lastAudioDetectedTime!);
        if (silenceDuration.inSeconds >= 2 && _hasAudioInput) {
          if (mounted) {
            setState(() {
              _hasAudioInput = false;
              _avModeText = '';
              debugPrint('🔇 Audio stopped - Switching to lip reading mode');
            });
          }
        }
      }
    });
  }

  Future<void> _startRecognition() async {
    // NEW: Check if server is connected before starting
    if (!_isServerConnected) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('❌ Cannot start: Server not connected. Enable server in settings.'),
            backgroundColor: Colors.red,
            duration: Duration(seconds: 3),
          ),
        );
      }
      return;
    }

    if (_speechService != null) {
      await _speechService!.start();
      
      setState(() {
        _isRecognizing = true;
        _frameCounter = 0;
        _audioChunksSent = 0;
        _lipFramesReceived = 0;
        _streamStartTime = DateTime.now();
        _lastAudioDetectedTime = DateTime.now();
        _lastFrameSentTime = null;  // NEW: Reset frame timing
        _status = 'Transcribing...';
        _simulatedLipText = '';
        _serverLipPrediction = '';
        _latestLipWord = '';
        _hasAudioInput = false;
        _hasLipDetection = false;
        _avModeText = '';
      });

      debugPrint('🎬 Starting Audio-Visual Transcription');
      _startRealtimeStreaming();
    }
  }

  Future<void> _stopRecognition() async {
    if (_speechService != null) {
      await _speechService!.stop();
      
      setState(() {
        _isRecognizing = false;
        _status = _isServerConnected ? 'AV Model Ready' : 'Ready (Connect Server)';
        _simulatedLipText = '';
        _latestLipWord = '';
        _avModeText = '';
        _avDelayTimer?.cancel();
        if (_interimText.isNotEmpty) {
           _transcribedText += '$_interimText\n';
           _interimText = '';
           _scrollToBottom();
        }
      });

      debugPrint('🛑 Stopping transcription');
      _stopRealtimeStreaming();
    }
  }

  void _startRealtimeStreaming() {
    _socket?.emit('start_session', {
      'session_id': DateTime.now().millisecondsSinceEpoch.toString(),
    });

    _startAudioStream();

    if (_isCameraInitialized && _cameraController != null) {
      _startVideoStream();
    }
  }

  void _startAudioStream() async {
    try {
      if (_audioRecorder != null && await _audioRecorder!.hasPermission()) {
        final stream = await _audioRecorder!.startStream(
          const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            sampleRate: 16000,
            numChannels: 1,
          ),
        );

        _audioStreamSubscription = stream.listen((audioData) {
          if (_isRecognizing) {
            _audioBuffer.addAll(audioData);
            
            if (_audioBuffer.length >= 16000) {
              _sendAudioChunk(_audioBuffer.sublist(0, 16000));
              _audioBuffer = _audioBuffer.sublist(16000);
            }
          }
        });

        debugPrint('🎤 Audio streaming started');
      }
    } catch (e) {
      debugPrint('❌ Audio stream error: $e');
    }
  }

  void _sendAudioChunk(List<int> audioData) {
    if (_socket != null && _isServerConnected) {
      final base64Audio = base64Encode(Uint8List.fromList(audioData));
      
      _socket!.emit('audio_chunk', {
        'audio': base64Audio,
        'timestamp': DateTime.now().toIso8601String(),
      });
      
      _audioChunksSent++;
    }
  }

  void _startVideoStream() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      debugPrint('❌ Camera controller not ready');
      return;
    }

    debugPrint('📸 Starting camera stream at $_targetFrameRate fps');
    
    try {
      await _cameraController!.startImageStream((CameraImage image) async {
        if (_isProcessingFrame || !_isRecognizing || !_isServerConnected) {
          return;
        }
        
        // NEW: Frame rate throttling - only send frames at 30fps
        final now = DateTime.now();
        if (_lastFrameSentTime != null) {
          final timeSinceLastFrame = now.difference(_lastFrameSentTime!).inMilliseconds;
          final minFrameInterval = (1000 / _targetFrameRate).round(); // ~33ms for 30fps
          
          if (timeSinceLastFrame < minFrameInterval) {
            return; // Skip this frame to maintain 30fps
          }
        }
        
        _isProcessingFrame = true;
        _lastFrameSentTime = now;
        
        try {
          final jpegBytes = await _convertCameraImageToJpeg(image);
          
          if (jpegBytes != null) {
            _sendVideoFrame(jpegBytes);
            _frameCounter++;
          }
        } catch (e) {
          debugPrint('❌ Frame processing error: $e');
        } finally {
          _isProcessingFrame = false;
        }
      });
      
      debugPrint('✅ Image stream started');
    } catch (e) {
      debugPrint('❌ Failed to start image stream: $e');
    }
  }

  void _sendVideoFrame(Uint8List frameData) {
    if (_socket != null && _isServerConnected) {
      final base64Frame = base64Encode(frameData);
      
      _socket!.emit('video_frame', {
        'frame': base64Frame,
        'timestamp': DateTime.now().toIso8601String(),
      });
    }
  }

  Future<Uint8List?> _convertCameraImageToJpeg(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;
      
      img.Image? convertedImage;
      
      if (image.format.group == ImageFormatGroup.yuv420) {
        final yPlane = image.planes[0];
        final uPlane = image.planes[1];
        final vPlane = image.planes[2];
        
        convertedImage = img.Image(width: width, height: height);
        
        final int uvRowStride = uPlane.bytesPerRow;
        final int uvPixelStride = uPlane.bytesPerPixel ?? 1;
        
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int yIndex = y * yPlane.bytesPerRow + x;
            final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;
            
            if (yIndex < yPlane.bytes.length && 
                uvIndex < uPlane.bytes.length && 
                uvIndex < vPlane.bytes.length) {
              final int yValue = yPlane.bytes[yIndex];
              final int uValue = uPlane.bytes[uvIndex];
              final int vValue = vPlane.bytes[uvIndex];
              
              int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
              int g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).round().clamp(0, 255);
              int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);
              
              convertedImage.setPixelRgba(x, y, r, g, b, 255);
            }
          }
        }
      } else if (image.format.group == ImageFormatGroup.jpeg) {
        return image.planes[0].bytes;
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        convertedImage = img.Image(width: width, height: height);
        final bytes = image.planes[0].bytes;
        
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int index = (y * width + x) * 4;
            if (index + 3 < bytes.length) {
              final int b = bytes[index];
              final int g = bytes[index + 1];
              final int r = bytes[index + 2];
              final int a = bytes[index + 3];
              convertedImage.setPixelRgba(x, y, r, g, b, a);
            }
          }
        }
      }
      
      if (convertedImage == null) {
        return null;
      }
      
      return Uint8List.fromList(img.encodeJpg(convertedImage, quality: 85));
      
    } catch (e) {
      debugPrint('❌ Image conversion error: $e');
      return null;
    }
  }

  void _stopRealtimeStreaming() async {
    if (_cameraController != null && _cameraController!.value.isStreamingImages) {
      try {
        await _cameraController!.stopImageStream();
        debugPrint('✅ Camera stream stopped');
      } catch (e) {
        debugPrint('⚠️ Error stopping camera stream: $e');
      }
    }
    
    if (_audioRecorder != null) {
      await _audioRecorder!.stop();
    }
    
    await _audioStreamSubscription?.cancel();
    _audioStreamSubscription = null;
    _audioBuffer.clear();

    _socket?.emit('end_session', {
      'session_id': DateTime.now().millisecondsSinceEpoch.toString(),
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // NEW: Helper function to determine what text to display based on current mode
  String _getCurrentDisplayText(bool hasAudio, bool hasLips) {
    if (hasAudio && hasLips) {
      // Audio-Visual mode: show AV text with delay
      return _avModeText.isEmpty ? '...Processing AV...' : _avModeText;
    } else if (hasAudio && !hasLips) {
      // Audio-only mode: show Vosk interim text
      return _interimText.isEmpty ? '...Listening...' : _interimText;
    } else if (!hasAudio && hasLips) {
      // Lip reading mode: show server lip prediction
      return _latestLipWord.isEmpty ? 'Reading lips...' : _latestLipWord;
    } else {
      // No inputs detected
      return 'Waiting for input...';
    }
  }

  void _showServerConfigDialog() {
    final ipController = TextEditingController(text: _serverIp);
    bool tempConnected = _isServerConnected;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Server Configuration'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: ipController,
                  decoration: const InputDecoration(
                    labelText: 'Server IP Address',
                    hintText: '192.168.254.104',
                    prefixIcon: Icon(Icons.computer),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
                const SizedBox(height: 16),
                SwitchListTile(
                  title: const Text('Server Connection'),
                  subtitle: Text(
                    tempConnected 
                      ? 'CONNECTED - Lip Preview Active' 
                      : 'DISCONNECTED - No Preview',
                    style: TextStyle(
                      fontSize: 12, 
                      color: tempConnected ? Colors.green : Colors.red,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  value: tempConnected,
                  onChanged: (value) {
                    setDialogState(() => tempConnected = value);
                  },
                ),
                const SizedBox(height: 8),
                if (_cameras != null && _cameras!.isNotEmpty) ...[
                  const Divider(),
                  const Text(
                    'Select Camera:',
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  ..._cameras!.asMap().entries.map((entry) {
                    final index = entry.key;
                    final camera = entry.value;
                    final isSelected = _selectedCameraIndex == index;
                    final isOBS = camera.name.toLowerCase().contains('obs') || 
                                   camera.name.toLowerCase().contains('virtual');
                    
                    return RadioListTile<int>(
                      dense: true,
                      title: Text(
                        camera.name,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: isOBS ? FontWeight.bold : FontWeight.normal,
                          color: isOBS ? Colors.green : Colors.black,
                        ),
                      ),
                      subtitle: Text(
                        '${camera.lensDirection}${isOBS ? ' 🔹 Recommended' : ''}',
                        style: TextStyle(fontSize: 10, color: Colors.grey[600]),
                      ),
                      value: index,
                      groupValue: _selectedCameraIndex,
                      onChanged: (value) {
                        setDialogState(() {
                          _selectedCameraIndex = value!;
                        });
                      },
                    );
                  }).toList(),
                ],
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Transcription Mode:',
                        style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        '🔊 Primary: Audio (when detected)',
                        style: TextStyle(fontSize: 10),
                      ),
                      const Text(
                        '👄 Fallback: Lip Reading (when silent)',
                        style: TextStyle(fontSize: 10),
                      ),
                      const Text(
                        '🎬 AV Mode: Both audio + video (enhanced)',
                        style: TextStyle(fontSize: 10),
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        '⚠️ Server connection required to start',
                        style: TextStyle(fontSize: 9, color: Colors.red, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton.icon(
              onPressed: () async {
                final newIp = ipController.text.trim();
                final ipChanged = newIp != _serverIp;
                
                setState(() {
                  _serverIp = newIp;
                  
                  if (tempConnected && !_isServerConnected) {
                    _connectToServer();
                  } else if (!tempConnected && _isServerConnected) {
                    _disconnectFromServer();
                  } else if (tempConnected && ipChanged) {
                    _disconnectFromServer();
                    Future.delayed(const Duration(milliseconds: 500), () {
                      _connectToServer();
                    });
                  }
                  
                  _cameraController?.dispose();
                  _initializeCamera();
                });
                
                await _saveServerConfig();
                Navigator.pop(context);
                
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      tempConnected 
                        ? 'Connecting to server: $_serverIp:$_serverPort'
                        : 'Disconnected',
                    ),
                    duration: const Duration(seconds: 2),
                  ),
                );
              },
              icon: const Icon(Icons.save),
              label: const Text('Save & Apply'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _audioSilenceTimer?.cancel();
    _avDelayTimer?.cancel();
    _audioStreamSubscription?.cancel();
    _speechService?.stop();
    _speechService?.dispose();
    _model?.dispose();
    _scrollController.dispose();
    _audioRecorder?.dispose();
    _cameraController?.dispose();
    _socket?.disconnect();
    _socket?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Color boxBackgroundColor = _displayTextColor == Colors.white ? Colors.black : Colors.white;

    final BoxDecoration displayBoxDecoration = BoxDecoration(
      color: boxBackgroundColor, 
      borderRadius: BorderRadius.circular(15),
      boxShadow: const [
        BoxShadow(
          color: Color.fromARGB(25, 0, 0, 0),
          spreadRadius: 2,
          blurRadius: 5,
          offset: Offset(0, 3),
        ),
      ],
    );

    final TextStyle liveTextStyle = TextStyle(
      fontFamily: 'Manrope',
      fontSize: _displayTextSize,
      color: _displayTextColor,
      fontWeight: FontWeight.normal,
    );

    final TextStyle historyTextStyle = TextStyle(
      fontFamily: 'Manrope',
      fontSize: 12.0, 
      color: _displayTextColor,
      fontWeight: FontWeight.normal,
    );

    // NEW: Use server's audio detection status
    final bool hasAudioActivity = _hasAudioInput;
    final bool hasLipActivity = _hasLipDetection;
    
    // Determine current mode: Audio-Visual, Audio-only, Lip Reading, or None
    String currentMode = 'None';
    Color modeColor = Colors.grey;
    IconData modeIcon = Icons.circle;
    
    if (hasAudioActivity && hasLipActivity) {
      currentMode = '🎬 Audio-Visual';
      modeColor = Colors.purple;
      modeIcon = Icons.multitrack_audio;
    } else if (hasAudioActivity && !hasLipActivity) {
      currentMode = '🔊 Audio Model';
      modeColor = Colors.blue;
      modeIcon = Icons.mic;
    } else if (!hasAudioActivity && hasLipActivity) {
      currentMode = '👄 Lip Reading';
      modeColor = Colors.orange;
      modeIcon = Icons.visibility;
    } else {
      currentMode = '⏸️ Waiting';
      modeColor = Colors.grey;
      modeIcon = Icons.pause_circle_outline;
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF2E3F8),
      appBar: AppBar(
        title: const Text(
          'Audio-Visual Transcription',
          style: TextStyle(fontFamily: 'Manrope', color: Color(0xFF49225B), fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Color(0xFF49225B)),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Row(
              children: [
                Icon(
                  _isServerConnected ? Icons.cloud_done : Icons.cloud_off,
                  color: _isServerConnected ? Colors.green : Colors.grey,
                  size: 20,
                ),
                if (_isCameraInitialized)
                  const Padding(
                    padding: EdgeInsets.only(left: 4),
                    child: Icon(
                      Icons.videocam,
                      color: Colors.green,
                      size: 18,
                    ),
                  ),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showServerConfigDialog,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text(
              'Status: $_status', 
              style: TextStyle(
                fontFamily: 'Manrope',
                fontSize: 14,
                color: _status.contains('Error') || _status.contains('Disconnected') 
                    ? Colors.redAccent 
                    : const Color(0xFF49225B),
                fontWeight: FontWeight.bold
              ),
              textAlign: TextAlign.center,
            ),
            
            if (_isRecognizing) ...[
              const SizedBox(height: 4),
              Text(
                'Mode: $currentMode',
                style: TextStyle(
                  fontFamily: 'Manrope',
                  fontSize: 11,
                  color: modeColor,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 2),
              Text(
                '📊 Frames: $_frameCounter | Audio: $_audioChunksSent chunks | Lip Previews: $_lipFramesReceived',
                style: const TextStyle(
                  fontFamily: 'Manrope',
                  fontSize: 9,
                  color: Colors.green,
                ),
                textAlign: TextAlign.center,
              ),
            ],
            const SizedBox(height: 16),
            
            // LIP PREVIEW SECTION (if server connected)
            if (_lipPreviewImage != null && _isServerConnected) ...[
              Container(
                height: 100,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.green, width: 2),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      '👄 Live Lip Region Preview',
                      style: TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Image.memory(
                      _lipPreviewImage!,
                      fit: BoxFit.contain,
                      height: 70,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ],
            
            // LIVE TRANSCRIPTION BOX
            Expanded(
              flex: 2, 
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: displayBoxDecoration.copyWith(
                  border: _isRecognizing 
                      ? Border.all(
                          color: modeColor,
                          width: 2
                        )
                      : null,
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          _isRecognizing 
                            ? '$currentMode Active'
                            : 'Waiting to Start',
                          style: TextStyle(
                            fontSize: 10,
                            color: !_isRecognizing ? Colors.grey : modeColor,
                            fontWeight: FontWeight.bold
                          ),
                        ),
                        if (_isRecognizing)
                          Padding(
                            padding: const EdgeInsets.only(left: 4),
                            child: Icon(
                              modeIcon,
                              size: 12,
                              color: modeColor,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Expanded(
                      child: Align(
                        alignment: _displayContainerAlignment, 
                        child: SingleChildScrollView(
                          child: Text(
                            !_isRecognizing
                                ? 'Press START...'
                                : _getCurrentDisplayText(hasAudioActivity, hasLipActivity),
                            style: liveTextStyle.copyWith(
                              color: !_isRecognizing ? Colors.grey : modeColor
                            ),
                            textAlign: _displayTextAlignment, 
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 12),
            const Text(
              'Transcript History:',
              style: TextStyle(
                fontSize: 14, 
                fontWeight: FontWeight.bold, 
                color: Color(0xFF49225B),
                fontFamily: 'Manrope'
              ),
            ),
            const SizedBox(height: 8), 
            
            // HISTORY BOX
            Expanded(
              flex: 1, 
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: displayBoxDecoration,
                child: SingleChildScrollView(
                  controller: _scrollController,
                  child: Text(
                    _transcribedText.isEmpty ? 'Full transcription appears here.' : _transcribedText,
                    style: historyTextStyle.copyWith(
                       color: _transcribedText.isEmpty ? Colors.grey : _displayTextColor
                    ),
                    textAlign: TextAlign.left,
                  ),
                ),
              ),
            ),
            
            const SizedBox(height: 20),

            // START/STOP BUTTON - NEW: Disabled when not connected
            Center(
              child: SizedBox(
                width: 280, 
                height: 50,
                child: ElevatedButton.icon(
                  onPressed: _modelLoaded && _isServerConnected
                      ? (_isRecognizing ? _stopRecognition : _startRecognition)
                      : null,
                  icon: Icon(_isRecognizing ? Icons.stop : Icons.mic, color: Colors.white),
                  label: Text(
                    _isRecognizing ? 'STOP' : (_isServerConnected ? 'START TRANSCRIPTION' : 'CONNECT SERVER FIRST'),
                    style: const TextStyle(
                      fontFamily: 'Manrope', 
                      fontSize: 13, 
                      fontWeight: FontWeight.bold
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isRecognizing 
                        ? Colors.redAccent.shade700 
                        : (_modelLoaded && _isServerConnected ? const Color(0xFF763B8D) : Colors.grey),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    elevation: 5,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}