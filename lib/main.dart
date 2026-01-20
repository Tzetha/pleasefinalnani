import 'dart:io'; // Required for HttpOverrides
import 'package:flutter/material.dart';
import 'landing-page.dart'; // Import the landing-page.dart file
import 'main-interface.dart'; // Import the main-interface.dart file

// --- SSL BYPASS CLASS ---
// This tells the app to ignore certificate errors (common fix for Android dev)
class MyHttpOverrides extends HttpOverrides {
  @override
  HttpClient createHttpClient(SecurityContext? context) {
    return super.createHttpClient(context)
      ..badCertificateCallback = (X509Certificate cert, String host, int port) => true;
  }
}


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartSense Meta Quest 3',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const LoadingPage(), // Set LoadingPage as the initial screen
    );
  }
}

class LoadingPage extends StatefulWidget {
  const LoadingPage({super.key});

  @override
  State<LoadingPage> createState() => _LoadingPageState();
}

class _LoadingPageState extends State<LoadingPage> {
  @override
  void initState() {
    super.initState();
    // Simulate loading process
    Future.delayed(const Duration(seconds: 3), () {
      // After the loading time, navigate to the main interface
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const MainInterface()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return const LandingPage(); // Display your existing landing-page.dart content as the loading screen
  }
}