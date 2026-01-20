import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:record/record.dart'; 
import 'package:web_socket_channel/io.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart'; // Import for loading settings

// --- CONFIGURATION ---
// Ensure this matches the output of: gcloud run services describe ...
const String websocketUrl = 'wss://quest-transcriber-523298308672.asia-southeast1.run.app'; 
const int sampleRate = 16000;
const int channels = 1; // Mono
// --- END CONFIGURATION ---

class TranscriptionScreen extends StatefulWidget {
  const TranscriptionScreen({super.key});

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen> {
  IOWebSocketChannel? _channel;
  StreamSubscription<Uint8List>? _micSubscription;
  final AudioRecorder _audioRecorder = AudioRecorder(); 
  
  // Controller for auto-scrolling history
  final ScrollController _scrollController = ScrollController();

  String _status = 'Initializing...';
  String _transcribedText = ''; 
  String _interimText = ''; 

  // --- Display Settings State ---
  double _displayTextSize = 16.0;
  Color _displayTextColor = Colors.black;
  TextAlign _displayTextAlignment = TextAlign.center;
  Alignment _displayContainerAlignment = Alignment.center; // Added for vertical/horizontal positioning

  @override
  void initState() {
    super.initState();
    _loadDisplaySettings(); // Load the saved visual preferences
    _requestPermissionsAndConnect();
  }

  // Load settings from Shared Preferences to match Display Configuration
  Future<void> _loadDisplaySettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      // Load Size
      _displayTextSize = prefs.getDouble('textSize') ?? 16.0;
      
      // Load Color
      int? colorVal = prefs.getInt('textColorValue');
      _displayTextColor = colorVal != null ? Color(colorVal) : Colors.black;

      // Load Alignment String
      String alignString = prefs.getString('textAlignment') ?? 'center';

      // 1. Determine TextAlignment (Justification within the text block)
      if (alignString.contains('Start')) {
        _displayTextAlignment = TextAlign.left;
      } else if (alignString.contains('End')) {
        _displayTextAlignment = TextAlign.right;
      } else {
        _displayTextAlignment = TextAlign.center;
      }

      // 2. Determine Container Alignment (Position of the text block in the box)
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

  Future<void> _requestPermissionsAndConnect() async {
    debugPrint("🔍 Checking permissions...");
    setState(() => _status = 'Requesting Mic Permission...');
    
    var status = await Permission.microphone.request();
    
    if (status.isGranted) {
      debugPrint("✅ Permission Granted. Connecting to server...");
      _connectWebSocket();
    } else {
      debugPrint("❌ Permission Denied.");
      setState(() => _status = 'Mic Permission Denied. Check Settings.');
      if (status.isPermanentlyDenied) {
        openAppSettings();
      }
    }
  }

  void _connectWebSocket() {
    try {
      debugPrint("🔵 Connecting to: $websocketUrl");
      _channel = IOWebSocketChannel.connect(Uri.parse(websocketUrl));
      
      setState(() => _status = 'Connecting...');
      
      _channel!.stream.listen(
        (message) {
          debugPrint("⬇️ RECEIVED: $message");
          if (!_status.contains('Connected')) {
             setState(() => _status = 'Connected & Transcribing');
          }
          _handleTranscription(message.toString());
        },
        onDone: () {
          debugPrint("🔶 WebSocket Closed by Server");
          setState(() {
            _status = 'Disconnected (Server Closed)';
            _stopStreaming();
          });
        },
        onError: (error) {
          debugPrint("❌ WebSocket Error: $error");
          setState(() {
            _status = 'Connection Error. Retrying...';
            _stopStreaming();
            Future.delayed(const Duration(seconds: 5), _connectWebSocket);
          });
        },
      );

    } catch (e) {
      debugPrint("❌ Connection Failed Exception: $e");
      setState(() => _status = 'Failed to connect: $e');
      Future.delayed(const Duration(seconds: 5), _connectWebSocket);
    }
  }

  void _handleTranscription(String message) {
    if (!mounted) return;

    // Filter out system messages
    if (message.contains("Vosk Ready")) {
      debugPrint("ℹ️ Ignored System Message: $message");
      return; 
    }

    bool isFinal = message.endsWith('.') || message.endsWith('?') || message.endsWith('!');
    
    setState(() {
      if (isFinal) {
        _transcribedText += '$message\n'; 
        _interimText = ''; 
        _scrollToBottom(); 
      } else {
        _interimText = message; 
      }
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

  Future<void> _startStreaming() async {
    if (_channel == null) {
      debugPrint("⚠️ WebSocket null, reconnecting...");
      _connectWebSocket();
      return;
    }

    if (await _audioRecorder.isRecording()) {
      return;
    }

    try {
      debugPrint("🎙️ Initializing Mic Stream...");
      
      final micStream = await _audioRecorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits, 
          sampleRate: sampleRate,
          numChannels: channels,
        ),
      );

      _micSubscription = micStream.listen((audioChunk) {
        try {
          _channel!.sink.add(audioChunk); 
        } catch (e) {
          debugPrint("❌ Error sending bytes: $e");
        }
      }, onError: (e) {
        debugPrint("❌ Mic Stream Error: $e");
        setState(() => _status = 'Mic Stream Error: $e');
        _stopStreaming();
      });

      setState(() {
        _status = 'Streaming Audio...';
        _transcribedText = ''; 
        _interimText = '';
        _channel!.sink.add('START_STREAMING'); 
      });
      debugPrint("✅ Audio Streaming Started");

    } catch (e) {
      debugPrint("❌ Start Stream Exception: $e");
      setState(() => _status = 'Audio setup failed: $e');
      _stopStreaming();
    }
  }

  void _stopStreaming() async {
    debugPrint("🛑 Stopping Stream...");
    _micSubscription?.cancel();
    _micSubscription = null;
    await _audioRecorder.stop();
    
    try {
      if (_channel != null) {
        _channel!.sink.add('STOP_STREAMING'); 
      }
    } catch (e) {
      // Ignore errors sending stop
    }
    
    if (mounted) {
      setState(() {
        _status = 'Ready to Start';
        if (_interimText.isNotEmpty) {
          _transcribedText += '$_interimText (Stopped)\n';
          _interimText = '';
          _scrollToBottom();
        }
      });
    }
  }
  
  @override
  void dispose() {
    _stopStreaming();
    _channel?.sink.close();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isStreaming = _micSubscription != null; 
    final isConnected = !_status.contains('Error') && !_status.contains('Disconnected');

    // Dynamic background color: Black if text is white, otherwise White
    final Color boxBackgroundColor = _displayTextColor == Colors.white ? Colors.black : Colors.white;

    final BoxDecoration displayBoxDecoration = BoxDecoration(
      color: boxBackgroundColor, 
      borderRadius: BorderRadius.circular(15),
      boxShadow: [
        BoxShadow(
          color: const Color.fromARGB(25, 0, 0, 0),
          spreadRadius: 2,
          blurRadius: 5,
          offset: const Offset(0, 3),
        ),
      ],
    );

    // Style for LIVE text (Follows user configuration)
    final TextStyle liveTextStyle = TextStyle(
      fontFamily: 'Manrope',
      fontSize: _displayTextSize,
      color: _displayTextColor,
      fontWeight: FontWeight.normal,
    );

    // Style for HISTORY text (Fixed size 12, user color)
    final TextStyle historyTextStyle = TextStyle(
      fontFamily: 'Manrope',
      fontSize: 12.0, // Fixed size as requested
      color: _displayTextColor,
      fontWeight: FontWeight.normal,
    );

    return Scaffold(
      backgroundColor: const Color(0xFFF2E3F8),
      appBar: AppBar(
        title: const Text(
          'Quest 3 Cloud Transcription',
          style: TextStyle(fontFamily: 'Manrope', color: Color(0xFF49225B), fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Color(0xFF49225B)),
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
                color: _status.contains('Error') || _status.contains('Denied') 
                    ? Colors.redAccent 
                    : const Color(0xFF49225B),
                fontWeight: FontWeight.bold
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            
            // --- LIVE LISTENING BOX ---
            Expanded(
              flex: 3, 
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: displayBoxDecoration.copyWith(
                  border: isStreaming 
                      ? Border.all(color: const Color(0xFF763B8D), width: 2) 
                      : null,
                ),
                // Use Align to position the ScrollView inside the Box based on configuration
                child: Align(
                  alignment: _displayContainerAlignment, 
                  child: SingleChildScrollView(
                    child: Text(
                      _interimText.isEmpty && !isStreaming
                          ? 'Press START to speak...'
                          : _interimText.isEmpty
                              ? '...Listening...'
                              : _interimText,
                      style: liveTextStyle.copyWith(
                        color: (_interimText.isEmpty && !isStreaming) 
                            ? Colors.grey 
                            : _displayTextColor
                      ),
                      textAlign: _displayTextAlignment, // Justification
                    ),
                  ),
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            const Text(
              'Transcript History:',
              style: TextStyle(
                fontSize: 16, 
                fontWeight: FontWeight.bold, 
                color: Color(0xFF49225B),
                fontFamily: 'Manrope'
              ),
            ),
            const SizedBox(height: 8),
            
            // --- HISTORY BOX ---
            Expanded(
              flex: 1, 
              child: Container(
                padding: const EdgeInsets.all(20),
                width: double.infinity,
                decoration: displayBoxDecoration,
                child: SingleChildScrollView(
                  controller: _scrollController,
                  child: Text(
                    _transcribedText.isEmpty ? 'The full transcription will appear here.' : _transcribedText,
                    style: historyTextStyle.copyWith(
                       color: _transcribedText.isEmpty ? Colors.grey : _displayTextColor
                    ),
                    textAlign: TextAlign.center, // Forced Center Alignment as requested
                  ),
                ),
              ),
            ),
            
            const SizedBox(height: 20),

            Center(
              child: SizedBox(
                width: 280, 
                height: 50,
                child: ElevatedButton.icon(
                  onPressed: isStreaming ? _stopStreaming : isConnected ? _startStreaming : null,
                  icon: Icon(isStreaming ? Icons.stop : Icons.mic, color: Colors.white),
                  label: Text(
                    isStreaming ? 'STOP STREAMING' : 'START STREAMING',
                    style: const TextStyle(
                      fontFamily: 'Manrope', 
                      fontSize: 16, 
                      fontWeight: FontWeight.bold
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isStreaming 
                        ? Colors.redAccent.shade700 
                        : (isConnected ? const Color(0xFF763B8D) : Colors.grey),
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