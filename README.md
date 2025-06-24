# AI Image Generator

An interactive AI-powered image generation tool that uses Stable Diffusion to create images from text prompts. The application includes both a command-line interface and a graphical user interface.

## Features

- Interactive GUI with real-time status updates
- Support for multiple image orientations (Landscape, Portrait, Square)
- Progress tracking during image generation
- Animal-specific training capabilities
- High-quality image output with customizable settings
- Memory-efficient processing
- Supports both GPU and CPU processing

## System Requirements

### For GPU Pipeline (Recommended)
- NVIDIA GPU with 4GB VRAM or more
- NVIDIA drivers installed
- CUDA toolkit compatible with PyTorch
- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### For CPU Pipeline
- Any 64-bit processor
- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Note: CPU processing will be significantly slower

## Step-by-Step Installation Guide

### 1. Install Python
1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Choose "Customize Installation"
5. Select all optional features
6. Set installation path to a location without spaces (e.g., `C:\Python312`)

### 2. Install Required Build Tools
1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer
3. Select "Desktop development with C++"
4. Click Install
5. Wait for installation to complete (this may take a while)
6. Restart your computer

### 3. Install Rust (Required for some dependencies)
1. Download Rust installer from [rustup.rs](https://rustup.rs/)
2. Run `rustup-init.exe`
3. Choose option 1 for default installation
4. Wait for installation to complete
5. Open a new command prompt to ensure PATH is updated

### 4. Set Up Python Virtual Environment
```bash
# Create a new directory for your project
mkdir ai_image_generator
cd ai_image_generator

# Clone this repository (if using git)
git clone <repository-url> .

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
```

### 5. Install Python Dependencies

#### For GPU Pipeline (Recommended if you have NVIDIA GPU):
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install GPU dependencies
pip install -r requirements-gpu.txt
```

#### For CPU Pipeline:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install CPU dependencies
pip install -r requirements-cpu.txt
```

### 6. Verify Installation
```bash
# Test if everything is installed correctly
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
python -c "import diffusers; print(diffusers.__version__)"
```

## Usage Guide

### Using the Interactive GUI

1. Start the GUI:
```bash
python interactive_generator.py
```

2. The interface includes:
   - Image preview area (top)
   - Progress bar (middle)
   - Prompt input box (bottom)
   - Orientation selection (Landscape/Portrait/Square)
   - Generate button

3. To generate an image:
   - Type your prompt in the text box
   - Select desired orientation
   - Click "Generate Image" or press Enter
   - Wait for generation to complete
   - Generated image will appear in preview area
   - Images are automatically saved in the `outputs` folder

### Using Command Line Interface

1. Start the CLI version:
```bash
python image_generator.py
```

2. Follow the prompts to:
   - Enter your text prompt
   - Choose image orientation
   - Wait for generation

## Performance Notes

### GPU Pipeline
- Faster generation (typically 5-15 seconds per image)
- Better memory management
- Supports larger image sizes
- CUDA optimizations for better quality

### CPU Pipeline
- Slower generation (can take 1-5 minutes per image)
- More RAM usage
- Limited to smaller image sizes
- May need to reduce quality settings for better performance

## Troubleshooting Guide

### Common Installation Issues

1. **"Microsoft Visual C++ 14.0 or greater is required"**
   - Solution: Install Visual Studio Build Tools as described in Step 2

2. **"Rust not found" or "Cargo not found"**
   - Solution: Install Rust using rustup as described in Step 3
   - Open a new command prompt after installation

3. **CUDA/GPU Issues**
   - Ensure NVIDIA drivers are up to date
   - Try running on CPU if GPU is unavailable
   - Check CUDA compatibility with PyTorch version

4. **Import Errors**
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Try reinstalling problematic packages

5. **Memory Issues**
   - Close other applications
   - Try generating smaller images
   - Use CPU mode if GPU memory is insufficient

### Error Messages

- "CUDA out of memory": Reduce image size or batch size
- "Module not found": Check if all requirements are installed
- "Access denied": Run command prompt as administrator

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify system requirements
3. Ensure all dependencies are correctly installed
4. Check GPU compatibility and drivers

## License

[Your License Here]

## Credits

- Stable Diffusion by CompVis
- Hugging Face for model hosting and datasets
- NVIDIA for CUDA support 