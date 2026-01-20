import 'package:flutter/material.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key}); // Added the key parameter

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack( // Use Stack to overlay widgets
        alignment: Alignment.center, // Center the children of the Stack
        children: [
          GradientBackground(),
          Positioned(
            top: MediaQuery.of(context).size.height / 2 - 100, // Adjusted to move the image up more
            child: Image.asset(
              'assets/white-glasses.png',
              height: 140, // Increased the height of the image
            ),
          ),
          Positioned(
            top: MediaQuery.of(context).size.height / 2 - 10, // Adjusted to move the text down slightly
            child: Text(
              'SMARTSENSE',
              style: TextStyle(
                fontFamily: 'SmartSense',
                fontSize: 24,
                fontWeight: FontWeight.normal,
                color: Color(0xFFF2E3F8),
                letterSpacing: 5 * 0.01 * 24,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class GradientBackground extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Color(0xFFF2E3F8),
            Color(0xFF8E5197),
            Color(0xFF49225B),
          ],
          stops: [0.0, 0.65, 1.0],
        ),
      ),
    );
  }
}