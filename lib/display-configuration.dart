import 'package:flutter/material.dart';
import 'package:smartsense_quest3/main-interface.dart'; // Import the main interface
import 'package:shared_preferences/shared_preferences.dart'; // Import for local storage

class DisplayConfigurationScreen extends StatefulWidget {
  const DisplayConfigurationScreen({super.key});

  @override
  State<DisplayConfigurationScreen> createState() => _DisplayConfigurationScreenState();
}

class _DisplayConfigurationScreenState extends State<DisplayConfigurationScreen> {
  // Keys for SharedPreferences
  static const String _textSizeKey = 'textSize';
  static const String _alignmentKey = 'textAlignment';
  static const String _textColorKey = 'textColorValue';

  // Default constants for reset functionality
  // Range is 15 to 30, so 22.5 is the exact middle size.
  static const double _resetTextSize = 22.5; 
  static const AlignmentDirectional _resetAlignment = AlignmentDirectional.center;
  static const Color _resetTextColor = Colors.black;

  double _currentTextSize = 22.5; // Default text size
  AlignmentDirectional _currentPreviewAlignment = AlignmentDirectional.center;
  Color _currentTextColor = Colors.black; // Default text color

  @override
  void initState() {
    super.initState();
    _loadSettings(); // Load saved preferences when the screen starts
  }

  // --- Utility Functions for Alignment String Conversion ---

  // Converts AlignmentDirectional to a storable string
  String _alignmentToString(AlignmentDirectional alignment) {
    // We use a map for reliable conversion to string
    if (alignment == AlignmentDirectional.topStart) return 'topStart';
    if (alignment == AlignmentDirectional.topCenter) return 'topCenter';
    if (alignment == AlignmentDirectional.topEnd) return 'topEnd';
    if (alignment == AlignmentDirectional.centerStart) return 'centerStart';
    if (alignment == AlignmentDirectional.center) return 'center';
    if (alignment == AlignmentDirectional.centerEnd) return 'centerEnd';
    if (alignment == AlignmentDirectional.bottomStart) return 'bottomStart';
    if (alignment == AlignmentDirectional.bottomCenter) return 'bottomCenter';
    if (alignment == AlignmentDirectional.bottomEnd) return 'bottomEnd';
    return 'center'; // Default fallback
  }

  // Converts a stored string back to AlignmentDirectional
  AlignmentDirectional _stringToAlignment(String alignmentString) {
    switch (alignmentString) {
      case 'topStart': return AlignmentDirectional.topStart;
      case 'topCenter': return AlignmentDirectional.topCenter;
      case 'topEnd': return AlignmentDirectional.topEnd;
      case 'centerStart': return AlignmentDirectional.centerStart;
      case 'centerEnd': return AlignmentDirectional.centerEnd;
      case 'bottomStart': return AlignmentDirectional.bottomStart;
      case 'bottomCenter': return AlignmentDirectional.bottomCenter;
      case 'bottomEnd': return AlignmentDirectional.bottomEnd;
      case 'center':
      default:
        return AlignmentDirectional.center;
    }
  }

  // --- Persistence Functions ---

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    
    // Load text size
    double? loadedTextSize = prefs.getDouble(_textSizeKey);
    
    // Safety check: If loaded size is outside the new bounds (15-30), clamp it.
    // This prevents crashes for users who saved '10.0' previously.
    if (loadedTextSize != null) {
      if (loadedTextSize < 15.0) loadedTextSize = 15.0;
      if (loadedTextSize > 30.0) loadedTextSize = 30.0;
    }

    // Load text color (stored as an integer value)
    final loadedTextColorValue = prefs.getInt(_textColorKey);
    
    // Load alignment (stored as a string)
    final loadedAlignmentString = prefs.getString(_alignmentKey);

    setState(() {
      // Apply loaded values or use defaults if they don't exist
      _currentTextSize = loadedTextSize ?? 22.5;
      
      _currentTextColor = loadedTextColorValue != null
          ? Color(loadedTextColorValue)
          : Colors.black;
          
      _currentPreviewAlignment = loadedAlignmentString != null
          ? _stringToAlignment(loadedAlignmentString)
          : AlignmentDirectional.center;
    });
  }

  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    
    // 1. Save Text Size (Double)
    await prefs.setDouble(_textSizeKey, _currentTextSize);
    
    // 2. Save Text Alignment (Convert AlignmentDirectional to String)
    final alignmentString = _alignmentToString(_currentPreviewAlignment);
    await prefs.setString(_alignmentKey, alignmentString);
    
    // 3. Save Text Color (Convert Color to Int value)
    // Updated to use toARGB32() instead of .value to fix deprecation warning
    await prefs.setInt(_textColorKey, _currentTextColor.toARGB32());

    // Show confirmation message (This handles the confirmation for both Save and Reset)
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Settings saved successfully!', style: TextStyle(color: Colors.white)),
          backgroundColor: const Color(0xFF49225B),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _resetSettings() async {
    setState(() {
      _currentTextSize = _resetTextSize;
      _currentPreviewAlignment = _resetAlignment;
      _currentTextColor = _resetTextColor;
    });

    // Save the reset values to persist the change
    await _saveSettings();

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Settings reset to default!', style: TextStyle(color: Colors.white)),
          backgroundColor: Colors.orange, // Distinct color for reset confirmation
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }


  @override
  Widget build(BuildContext context) {
    // Determine the TextAlign for the Text widget based on _currentPreviewAlignment
    TextAlign textAlignmentForPreview;
    if (_currentPreviewAlignment == AlignmentDirectional.centerStart ||
        _currentPreviewAlignment == AlignmentDirectional.bottomStart ||
        _currentPreviewAlignment == AlignmentDirectional.topStart) {
      textAlignmentForPreview = TextAlign.left;
    } else if (_currentPreviewAlignment == AlignmentDirectional.centerEnd ||
        _currentPreviewAlignment == AlignmentDirectional.bottomEnd ||
        _currentPreviewAlignment == AlignmentDirectional.topEnd) {
      textAlignmentForPreview = TextAlign.right;
    } else {
      textAlignmentForPreview = TextAlign.center;
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF2E3F8), // Match background from image
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
            child: const Padding(
              padding: EdgeInsets.only(left: 16.0, right: 12.0),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.arrow_back_ios, color: Color(0xFF49225B), size: 16),
                  SizedBox(width: 4),
                  Text(
                    'Back',
                    style: TextStyle(
                      fontFamily: 'Manrope',
                      fontSize: 16,
                      color: Color(0xFF49225B),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        centerTitle: true,
        title: const SizedBox.shrink(), // Removed title as per image
      ),
      // --- WRAPPED BODY IN SINGLE CHILD SCROLL VIEW ---
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0), // Adjusted padding to match image
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch, // Stretch children horizontally
            children: <Widget>[
              const SizedBox(height: 16), // Initial spacing
              const Text(
                'DISPLAY CONFIGURATION',
                textAlign: TextAlign.center, // Aligned left as per image
                style: TextStyle(
                  fontFamily: 'SmartSense', // Assuming Manrope or similar bold font for title
                  fontSize: 26, // Adjusted font size
                  color: Color(0xFF49225B), // Adjusted color
                  fontWeight: FontWeight.bold, // Added bold weight
                ),
              ),
              const SizedBox(height: 20), // Spacing below title
              const Text(
                'Display Preview',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontFamily: 'Manrope', // Assuming Manrope or similar
                  fontSize: 14,
                  color: Color(0xFF49225B), // Adjust color if needed
                  fontWeight: FontWeight.bold, // Added bold weight
                ),
              ),
              const SizedBox(height: 10), // Spacing below preview label
              SizedBox(
                height: 120.0, // Fixed height for the preview box
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 10.0, horizontal: 10.0), // Adjust padding as needed within the fixed height
                  margin: const EdgeInsets.symmetric(horizontal: 10.0), // Reduced margin for wider box
                  decoration: BoxDecoration(
                    color: _currentTextColor == Colors.white ? Colors.black : Colors.white, // Change color to black if text is white
                    borderRadius: BorderRadius.circular(15),
                    boxShadow: [
                      BoxShadow(
                        // Using explicit ARGB constructor to fix deprecated warning
                        color: const Color.fromARGB(25, 0, 0, 0), 
                        spreadRadius: 2,
                        blurRadius: 5,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: Align(
                    alignment: _currentPreviewAlignment, // Use AlignmentDirectional for positioning
                    child: Text(
                      'This is what your\ntext will look like.',
                      textAlign: textAlignmentForPreview, // Use derived TextAlign for horizontal alignment of the text itself
                      style: TextStyle(
                        fontFamily: 'Manrope', // Use a default font for preview
                        fontSize: _currentTextSize,
                        color: _currentTextColor,
                        foreground: _currentTextColor == Colors.white
                            ? null // No foreground paint needed if box is black
                            : null,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20), // Spacing after preview
              _buildSettingSection(
                title: 'Text Size',
                content: Row(
                  children: [
                    const Text('A', style: TextStyle(fontSize: 10, color: Color(0xFF49225B), fontWeight: FontWeight.bold)), // Smaller 'A'
                    Expanded(
                      child: Slider(
                        value: _currentTextSize,
                        min: 15, // Updated MINIMUM
                        max: 30, // Updated MAXIMUM
                        divisions: 15, // Adjusted divisions for the new range (15 steps)
                        activeColor: const Color(0xFF49225B), // Adjust color
                        inactiveColor: const Color(0xFFD3B3E7), // Adjust color
                        onChanged: (double value) {
                          setState(() {
                            _currentTextSize = value;
                          });
                        },
                      ),
                    ),
                    const Text('A', style: TextStyle(fontSize: 20, color: Color(0xFF49225B), fontWeight: FontWeight.bold)), // Smaller 'A'
                  ],
                ),
              ),
              const SizedBox(height: 20), // Spacing between sections
              _buildSettingSection(
                title: 'Text Alignment',
                content: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly, // Distribute buttons evenly
                  children: [
                    _buildAlignmentButton(AlignmentDirectional.centerStart, 'assets/icons/tabler_align-box-left-middle.png', 'Middle-Left'),
                    _buildAlignmentButton(AlignmentDirectional.center, 'assets/icons/tabler_align-box-center-middle.png', 'Middle-Center'),
                    _buildAlignmentButton(AlignmentDirectional.centerEnd, 'assets/icons/tabler_align-box-right-middle.png', 'Middle-Right'),
                    _buildAlignmentButton(AlignmentDirectional.bottomStart, 'assets/icons/icon-park-outline_alignment-left-bottom.png', 'Bottom-Left'),
                    _buildAlignmentButton(AlignmentDirectional.bottomCenter, 'assets/icons/tabler_align-box-center-bottom.png', 'Bottom-Center'),
                    _buildAlignmentButton(AlignmentDirectional.bottomEnd, 'assets/icons/tabler_align-box-right-bottom.png', 'Bottom-Right'),
                  ],
                ),
              ),
              const SizedBox(height: 20), // Spacing between sections
              _buildSettingSection(
                title: 'Text Color',
                content: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildColorOption(Colors.black),
                    _buildColorOption(Colors.white),
                    _buildColorOption(Colors.green),
                    _buildColorOption(Colors.yellow),
                  ],
                ),
              ),
              const SizedBox(height: 20), // Spacing before buttons
      
              // --- Save and Reset Buttons ---
              Center( 
                child: Column(
                  children: [
                    SizedBox(
                      width: 150, 
                      height: 35, 
                      child: ElevatedButton(
                        onPressed: _saveSettings, // Call the save function
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF49225B), // Button background color
                          foregroundColor: Colors.white, // Text color
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10), // Rounded corners
                          ),
                          textStyle: const TextStyle(
                            fontFamily: 'Manrope',
                            fontWeight: FontWeight.bold,
                            fontSize: 14, // Smaller font size for the button text
                          ),
                        ),
                        child: const Text('Save Settings'),
                      ),
                    ),
                    const SizedBox(height: 10), // Spacing between buttons
                    SizedBox(
                      width: 170,
                      height: 35,
                      child: OutlinedButton( // Use OutlinedButton for secondary action
                        onPressed: _resetSettings, // Call the new reset function
                        style: OutlinedButton.styleFrom(
                          foregroundColor: const Color(0xFF49225B), // Text color
                          side: const BorderSide(color: Color(0xFF49225B), width: 2), // Border color
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                          textStyle: const TextStyle(
                            fontFamily: 'Manrope',
                            fontWeight: FontWeight.bold,
                            fontSize: 14,
                          ),
                        ),
                        child: const Text('Reset to Default'),
                      ),
                    ),
                  ],
                ),
              ),
              // --- End Buttons ---
      
              const SizedBox(height: 40), 
              
              const Padding(
                padding: EdgeInsets.only(bottom: 16.0),
                child: Text(
                  '© SMARTSENSE 2025',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontFamily: 'Manrope',
                    fontSize: 12,
                    color: Color(0xFF49225B),
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingSection({required String title, required Widget content}) {
    // NOTE: screenWidth removed as it was unused and the calculations rely on fixed padding/margin.
    final horizontalMargin = 10.0;
    // The padding for the title should align with the content's left edge
    final titleHorizontalPadding = horizontalMargin + 20.0; // 20.0 from parent padding + 10.0 for the box's margin

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: EdgeInsets.symmetric(horizontal: titleHorizontalPadding), // Adjusted padding for title
          child: Text(
            title,
            style: const TextStyle(
              fontFamily: 'Manrope',
              fontSize: 14,
              color: Color(0xFF49225B),
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10.0, vertical: 5.0), // Keep internal padding
          margin: EdgeInsets.symmetric(horizontal: horizontalMargin), // Adjusted margin for wider box
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(15),
            boxShadow: [ 
              BoxShadow(
                // Using explicit ARGB constructor to fix deprecated warning
                color: const Color.fromARGB(25, 0, 0, 0),
                spreadRadius: 2,
                blurRadius: 5,
                offset: const Offset(0, 3),
              ),
            ],
          ),
          child: content,
        ),
      ],
    );
  }

  Widget _buildAlignmentButton(AlignmentDirectional alignment, String iconPath, String tooltip) {
    bool isSelected = _currentPreviewAlignment == alignment;
    return GestureDetector(
      onTap: () {
        setState(() {
          _currentPreviewAlignment = alignment;
        });
      },
      child: Tooltip(
        message: tooltip,
        child: Container(
          width: 30, // Slightly reduced width to fit 6 buttons
          height: 30, // Slightly reduced height
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF49225B) : Colors.transparent, // Selected color
            borderRadius: BorderRadius.circular(8), // Slight rounding
            // Removed the border property to eliminate the outline
          ),
          child: Padding( // Added padding to center the icon
            padding: const EdgeInsets.all(4.0),
            child: Image.asset(
              iconPath,
              color: isSelected ? Colors.white : const Color(0xFF49225B), // Icon color
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildColorOption(Color color) {
    bool isSelected = _currentTextColor == color;
    return GestureDetector(
      onTap: () {
        setState(() {
          _currentTextColor = color;
        });
      },
      child: Container(
        width: 25, // Smaller size of color circle
        height: 25, // Smaller size of color circle
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: color,
          // Add a black border for white color to make it visible
          border: isSelected
              ? Border.all(color: const Color(0xFF49225B), width: 3) // Thicker border for selected
              : (color == Colors.white
                  ? Border.all(color: Colors.black, width: 1) // Black outline for white
                  : Border.all(color: Colors.grey.shade400, width: 1)), // Thin border for unselected
        ),
        child: isSelected
            ? const Icon(
                Icons.check,
                color: Colors.white, // Checkmark for selected
                size: 15, // Smaller checkmark size
              )
            : null,
      ),
    );
  }
}