import 'package:flutter/material.dart';
import 'package:smartsense_quest3/main-interface.dart'; // Import the main interface

class AboutUsScreen extends StatelessWidget {
  const AboutUsScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF2E3F8),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leadingWidth: 100,
        leading: Align(
          alignment: Alignment.centerLeft,
          child: InkWell(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => MainInterface()),
              );
            },
            child: Padding(
              padding: const EdgeInsets.only(left: 16.0, right: 12.0),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.arrow_back_ios,
                      color: Color(0xFF49225B), size: 16),
                  const SizedBox(width: 4),
                  Text(
                    'Back',
                    style: TextStyle(
                      fontFamily: 'Manrope',
                      fontSize: 16,
                      color: const Color(0xFF49225B),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        centerTitle: true,
        title: const SizedBox.shrink(),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40.0),
        child: SingleChildScrollView( // Added SingleChildScrollView for longer content
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const SizedBox(height: 20),
              const Text(
                'ABOUT US',
                style: TextStyle(
                  fontFamily: 'SmartSense',
                  fontSize: 26,
                  color: Color(0xFF49225B),
                ),
              ),
              const SizedBox(height: 16),
              RichText(
                text: const TextSpan(
                  style: TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 14,
                    color: Color(0xFF49225B),
                    height: 1.5,
                  ),
                  children: [
                    TextSpan(
                        text:
                            'We are Group 3 of '),
                    TextSpan(
                        text: 'Bachelor of Science in Computer Science (BSCS) 3-A (AI Specialization)',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    TextSpan(
                        text: ', a team of four computer science students dedicated to developing innovative solutions that leverage artificial intelligence for social good. Our thesis, '),
                    TextSpan(
                        text: 'SmartSense: Augmented Reality (AR) Glasses for Audio-Visual to Text Translation as an Assistive Tool for Deaf Individuals',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    TextSpan(
                        text: ', aims to bridge communication gaps for the deaf and hearing-impaired community by providing '),
                    TextSpan(
                        text: 'transcription, empowering them to engage and communicate more effectively.',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                'Development Team',
                style: TextStyle(
                  fontFamily: 'SmartSense',
                  fontSize: 22,
                  color: Color(0xFF49225B),
                ),
              ),
              const SizedBox(height: 16),
              Center(
                child: Column(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8.0),
                      child: Image.asset(
                        'assets/code3.png', // Replace with your image path
                        width: 100,
                        height: 100,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Ethan Gabriel Soncio',
                      style: TextStyle(
                        fontFamily: 'Manrope',
                        fontSize: 16,
                        color: Color(0xFF49225B),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Text(
                      'Lead Back- End Developer',
                      style: TextStyle(
                        fontFamily: 'Manrope',
                        fontSize: 14,
                        color: Color(0xFF49225B),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.computer), // Replace with your icon
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: const Icon(Icons.email), // Replace with your icon
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: Image.asset('assets/icons/linkedin.png'),
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                'Development Team', // Assuming another team member follows
                style: TextStyle(
                  fontFamily: 'SmartSense',
                  fontSize: 22,
                  color: Color(0xFF49225B),
                ),
              ),
              const SizedBox(height: 16),
              Center(
                child: Column(
                  children: [
                    // Replace with your actual image asset
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8.0),
                      child: Image.asset(
                        'assets/code3.png', // Replace with your image path
                        width: 100,
                        height: 100,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Chariz Dianne Falco',
                      style: TextStyle(
                        fontFamily: 'Manrope',
                        fontSize: 16,
                        color: Color(0xFF49225B),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Text(
                      'Lead Mobile App Developer',
                      style: TextStyle(
                        fontFamily: 'Manrope',
                        fontSize: 14,
                        color: Color(0xFF49225B),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.computer), // Replace with your icon
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: const Icon(Icons.email), // Replace with your icon
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: Image.asset('assets/icons/linkedin.png'), // Replace with your icon
                          color: const Color(0xFF49225B),
                          onPressed: () {
                            // Handle button press
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.only(bottom: 16.0),
        child: Text(
          '© SMARTSENSE 2025',
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontFamily: 'Manrope',
            fontSize: 12,
            color: Color(0xFF49225B),
            fontWeight: FontWeight.w400,
          ),
        ),
      ),
    );
  }
}