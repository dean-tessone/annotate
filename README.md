# AnnotateEZ

A robust image annotation tool for HDF5 datasets with dynamic image loading, intelligent caching, and flexible color management.

## Features

- **Dynamic Image Loading**: Loads images on-demand with intelligent caching to minimize memory usage
- **Flexible Color Management**: Multiple color schemes (default, pastel, vibrant, monochrome) with automatic color assignment
- **Memory Efficient**: Configurable cache size and LRU eviction for optimal performance
- **Interactive Interface**: Easy-to-use GUI with keyboard shortcuts and real-time updates
- **HDF5 Support**: Native support for HDF5 image datasets
- **Configurable**: Extensive configuration options through YAML files

## Prerequisites

- Python 3.7 or higher
- Conda (recommended) or pip

## Installation

### Option 1: Using Conda (Recommended)

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd annotateEZ
   ```

2. **Create a conda environment**:
   ```bash
   conda create -n annotateez python=3.9
   conda activate annotateez
   ```

3. **Install dependencies**:
   ```bash
   conda install -c conda-forge pyqt5 pandas numpy h5py pyyaml
   ```
   
   Or install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using pip only

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv annotateez_env
   source annotateez_env/bin/activate  # On Windows: annotateez_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. **Activate your environment**:
   ```bash
   conda activate annotateez  # or activate your virtual environment
   ```

2. **Run the application**:
   ```bash
   python annotateEZ.py
   ```

3. **Load your HDF5 dataset**:
   - Click the "Load" button
   - Select your HDF5 file
   - The application will automatically detect image and data keys

### Configuration

The application uses `config.yml` for configuration. Key settings include:

- **Color Scheme**: Choose from 'default', 'pastel', 'vibrant', or 'monochrome'
- **Cache Size**: Number of images to keep in memory (default: 100)
- **Grid Size**: Number of tiles per page (x_size × y_size)
- **Tile Size**: Size of each image tile in pixels

Example configuration:
```yaml
color_scheme: default
image_cache_size: 100
x_size: 15
y_size: 15
tile_size: 85
```

### Keyboard Shortcuts

- **Left/Right Arrow Keys**: Navigate between pages
- **Ctrl+Shift+C**: Clear image cache to free memory
- **Left Click**: Select/flag an image tile
- **Right Click**: Mark an image tile as junk

### HDF5 File Format

Your HDF5 file should contain:
- **Images dataset**: Multi-dimensional array with shape (n_images, height, width, channels)
- **Features dataset**: Pandas DataFrame with metadata and labels

Example HDF5 structure:
```
file.hdf5
├── images (dataset: shape=(1000, 64, 64, 4))
├── features (dataset: pandas DataFrame)
└── labels (dataset: optional, label names)
```

## Advanced Features

### Dynamic Color Management

The application automatically assigns colors to labels based on the selected color scheme:
- **Default**: Standard colors (red, blue, green, yellow, etc.)
- **Pastel**: Soft, muted colors
- **Vibrant**: Bright, saturated colors
- **Monochrome**: Grayscale variations

### Memory Management

- **Configurable Cache**: Adjust cache size based on available memory
- **LRU Eviction**: Automatically removes least recently used images
- **Preloading**: Intelligently preloads adjacent pages for smooth navigation

### Settings Interface

Access settings by running the application:
- Configure label names and colors
- Set output directory
- Adjust grid and tile sizes
- Change color schemes
- Modify cache settings

## Troubleshooting

### Common Issues

1. **"Images not found in input file"**:
   - Check that your HDF5 file contains the correct image key (default: 'images')
   - Verify the image dataset exists and is accessible

2. **Memory issues with large datasets**:
   - Reduce the `image_cache_size` in config.yml
   - Use Ctrl+Shift+C to manually clear the cache
   - Consider using a machine with more RAM

3. **Slow performance**:
   - Increase cache size if you have sufficient memory
   - Check that your HDF5 file is not corrupted
   - Ensure images are properly formatted (uint16 or uint8)

4. **Color display issues**:
   - Try different color schemes in settings
   - Check that your display supports the color format
   - Restart the application after changing color schemes

### Performance Tips

- **Optimal Cache Size**: Start with 50-100 images, adjust based on available memory
- **File Location**: Keep HDF5 files on fast storage (SSD recommended)
- **Image Format**: Use uint16 for better quality, uint8 for smaller files
- **Grid Size**: Larger grids show more images but may impact performance

## File Structure

```
annotateEZ/
├── annotateEZ.py          # Main application
├── config.yml             # Configuration file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── main.log              # Application log (created at runtime)
└── icons/                # UI icons (264 PNG files)
```

## Dependencies

- **PyQt5**: GUI framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **h5py**: HDF5 file handling
- **PyYAML**: Configuration file parsing

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the log file (`main.log`) for error details
3. [Add your support contact information here]
