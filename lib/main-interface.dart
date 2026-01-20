import 'package:flutter/material.dart';
import 'package:smartsense_quest3/display-configuration.dart';
import 'package:smartsense_quest3/about-us.dart';
import 'package:smartsense_quest3/startLocalTranslation.dart';
import 'startTranslation.dart'; // Import the main-interface.dart file

class MainInterface extends StatelessWidget {
  const MainInterface({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF2E3F8), // Background color (from the image)
      body: Stack( // Use Stack to allow overlapping widgets
        alignment: Alignment.center, // Center the children of the Stack
        children: [
          Positioned(
            top: 80, // Adjust this value to control the vertical position of the image
            child: Image.asset(
              'assets/purple-glasses.png', // Replace with your image path
              width: 150,
              height: 150,
              fit: BoxFit.contain, // Or BoxFit.cover, BoxFit.fitWidth, etc.
            ),
          ),
          Positioned(
            top: 180, // Adjust this value to control the vertical position of the text
            child: Text(
              'SMARTSENSE',
              style: const TextStyle(
                fontFamily: 'SmartSense',
                fontSize: 32, // Approximate size
                color: Color(0xFF49225B), // Primary purple color (from the image)
              ),
            ),
          ),
          
          // --- BUTTONS (Width increased to 280) ---

          Positioned(
            top: 260, 
            child: SizedBox(
              width: 280, // Increased width
              height: 45,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => LocalTranscriptionScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 14, 
                    fontWeight: FontWeight.bold, // Bold for first button
                  ),
                  padding: EdgeInsets.zero,
                ),
                child: Ink(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: const [
                        Color(0xFF763B8D),
                        Color(0xFF211027),
                      ],
                      stops: const [0.0, 1.0],
                    ),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Center(
                    child: Text(
                      'START LOCAL TRANSCRIPTION',
                      style: TextStyle(
                        color: Colors.white,
                        fontFamily: 'Manrope',
                        fontSize: 14, 
                        fontWeight: FontWeight.bold, 
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 321, 
            child: SizedBox(
              width: 280, // Increased width
              height: 45,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => TranscriptionScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  // Updated to use withValues() to fix deprecation warning
                  backgroundColor: const Color(0xFF49225B).withValues(alpha: 0.63),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 14, 
                    fontWeight: FontWeight.w600, // SemiBold
                  ),
                ),
                child: const Text('START CLOUD TRANSCRIPTION'),
              ),
            ),
          ),
          Positioned(
            top: 382, 
            child: SizedBox(
              width: 280, // Increased width
              height: 45,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => DisplayConfigurationScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  // Updated to use withValues() to fix deprecation warning
                  backgroundColor: const Color(0xFF49225B).withValues(alpha: 0.63),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 14, 
                    fontWeight: FontWeight.w600, // SemiBold
                  ),
                ),
                child: const Text('CONFIGURE DISPLAY'),
              ),
            ),
          ),
          Positioned(
            top: 443, 
            child: SizedBox(
              width: 280, // Increased width
              height: 45,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => AboutUsScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  // Updated to use withValues() to fix deprecation warning
                  backgroundColor: const Color(0xFF49225B).withValues(alpha: 0.63),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  textStyle: const TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 14, 
                    fontWeight: FontWeight.w600, // SemiBold
                  ),
                ),
                child: const Text('ABOUT US'),
              ),
            ),
          ),
          const Positioned(
            bottom: 16.0,
            child: Text(
              '© SMARTSENSE 2025',
              style: TextStyle(
                fontFamily: 'Manrope',
                fontSize: 10,
                color: Color(0xFF49225B),
                fontWeight: FontWeight.w400, 
              ),
            ),
          ),
        ],
      ),
    );
  }
}