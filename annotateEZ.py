from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
import sys
import os
import h5py
import yaml
import logging
from collections import OrderedDict
import colorsys
import random
# Input
images = []
df = pd.DataFrame()

# Constants:
config_path = 'config.yml'
log_path    = 'main.log'

# Logger setup
logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
console_format = logging.Formatter("[%(levelname)s] %(message)s")
c_handler.setFormatter(console_format)
c_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(c_handler)
f_handler = logging.FileHandler(filename=log_path, mode='w')
f_format = logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s")
f_handler.setFormatter(f_format)
f_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(f_handler)
logging.getLogger().setLevel(logging.DEBUG)


def channels2rgb8bit(image):
    "Convert 4 channel images to 8-bit RGB color images."
    assert(image.dtype == 'uint16')
    image = image.astype('float')
    if(len(image.shape) == 4):
        image[:, :, :, 0:3] = image[:, :, :, [1,2,0]]
        if(image.shape[3] > 3):
            image = image[:, :, :, 0:3] + np.expand_dims(image[:, :, :, 3], 3)
        
    elif(len(image.shape) == 3):
        image[:, :, 0:3] = image[:, :, [1, 2, 0]]
        if(image.shape[2] > 3):
            image = image[:, :, 0:3] + np.expand_dims(image[:, :, 3], 2)
        
    image[image > 65535] = 65535
    image = (image // 256).astype('uint8')
    return(image)


class ImageCacheManager:
    """Manages dynamic loading and caching of images for memory efficiency."""
    
    def __init__(self, file_path, image_key, cache_size=50):
        self.file_path = file_path
        self.image_key = image_key
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.file_handle = None
        self.image_dataset = None
        self.image_shape = None
        self.n_events = 0
        self.selected_channels = ['composite']  # Default to composite view
        
    def open_file(self):
        """Open the HDF5 file and get image dataset reference."""
        if self.file_handle is None:
            self.file_handle = h5py.File(self.file_path, 'r')
            self.image_dataset = self.file_handle[self.image_key]
            self.image_shape = self.image_dataset.shape
            self.n_events = self.image_shape[0]
            logger.info(f"Opened file with {self.n_events} images")
    
    def close_file(self):
        """Close the HDF5 file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.image_dataset = None
    
    def _to_rgb888(self, image_data, channel_mode='composite'):
        """Convert various image shapes/dtypes to contiguous uint8 RGB (H, W, 3)."""
        if channel_mode == 'composite':
            # Handle 2D grayscale
            if image_data.ndim == 2:
                rgb = np.stack([image_data, image_data, image_data], axis=2)
            elif image_data.ndim == 3:
                h, w, c = image_data.shape
                # Convert uint16 using existing pipeline (which reorders channels)
                if image_data.dtype == np.uint16:
                    rgb = channels2rgb8bit(image_data.reshape(1, h, w, c))[0]
                else:
                    # For uint8 or others, reorder channels similar to channels2rgb8bit
                    tmp = image_data.astype(np.uint8, copy=False)
                    if c >= 3:
                        # swap [1,2,0]
                        tmp = tmp[:, :, [1, 2, 0]]
                        if c > 3:
                            # approximate adding alpha channel then clip
                            add = tmp[:, :, 0:3] + np.expand_dims(image_data[:, :, 3].astype(np.uint8), 2)
                            tmp = np.clip(add, 0, 255).astype(np.uint8)
                        rgb = tmp[:, :, 0:3]
                    else:
                        # replicate channels if less than 3
                        rgb = np.repeat(tmp, 3, axis=2)
            else:
                # Unexpected shape; return black
                rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
        else:
            # Single channel mode
            if image_data.ndim == 2:
                # Already 2D, convert to RGB
                rgb = np.stack([image_data, image_data, image_data], axis=2)
            elif image_data.ndim == 3:
                h, w, c = image_data.shape
                try:
                    channel_idx = int(channel_mode)
                    if 0 <= channel_idx < c:
                        # Extract specific channel
                        single_channel = image_data[:, :, channel_idx]
                        if single_channel.dtype == np.uint16:
                            single_channel = (single_channel // 256).astype(np.uint8)
                        rgb = np.stack([single_channel, single_channel, single_channel], axis=2)
                    else:
                        # Channel out of range, return black
                        rgb = np.zeros((h, w, 3), dtype=np.uint8)
                except (ValueError, IndexError):
                    # Invalid channel, return black
                    rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
            else:
                # Unexpected shape; return black
                rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
        
        # Ensure contiguous
        rgb = np.ascontiguousarray(rgb)
        return rgb

    def set_selected_channels(self, channels):
        """Set which channels to display."""
        self.selected_channels = channels
        # Clear cache when channel selection changes
        self.cache.clear()

    def get_image(self, image_id, channel_mode='composite'):
        """Get image by ID with caching."""
        # Ensure file is open and n_events initialized
        self.open_file()
        if image_id >= self.n_events:
            return None
            
        # Create cache key that includes channel mode
        cache_key = f"{image_id}_{channel_mode}"
        
        # Check cache first
        if cache_key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # Load from file
        image_data = self.image_dataset[image_id]
        
        # Convert to contiguous RGB888 with specified channel mode
        image_data = self._to_rgb888(image_data, channel_mode)
        
        # Add to cache
        self.cache[cache_key] = image_data
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return image_data
    
    def preload_range(self, start_id, end_id):
        """Preload a range of images for better performance."""
        for i in range(start_id, min(end_id, self.n_events)):
            if i not in self.cache:
                self.get_image(i)
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self.cache.clear()
        logger.info("Image cache cleared")


class ColorManager:
    """Manages color assignment for labels (default scheme only)."""
    
    def __init__(self):
        self.color_map = {}
        self.used_colors = set()
        self.color_schemes = {
            'default': ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        }
        self.current_scheme = 'default'
    
    def get_color_for_label(self, label_id, label_name):
        """Get or assign a color for a label."""
        if label_id in self.color_map:
            return self.color_map[label_id]
        
        # Special case: label 0 (class 0) always gets black
        if label_id == 0:
            color = 'black'
            self.color_map[label_id] = color
            self.used_colors.add(color)
            return color
        
        # Get available colors from default scheme
        available_colors = [c for c in self.color_schemes['default'] 
                           if c not in self.used_colors]
        
        if not available_colors:
            # Generate a random color if we run out
            color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
        else:
            color = available_colors[0]
        
        self.color_map[label_id] = color
        self.used_colors.add(color)
        return color
    
    # No-op: color scheme changes are disabled; default is always used
    
    def reset_colors(self):
        """Reset all assigned colors."""
        self.color_map.clear()
        self.used_colors.clear()
        logger.info("Color assignments reset")
    
    def get_qt_color(self, color_string):
        """Convert color string to Qt color."""
        if color_string.startswith('rgb('):
            # Parse RGB string
            rgb_values = color_string[4:-1].split(', ')
            r, g, b = map(int, rgb_values)
            return QColor(r, g, b)
        else:
            # Handle named colors
            color_map = {
                'red': Qt.red,
                'blue': Qt.blue,
                'green': Qt.green,
                'yellow': Qt.yellow,
                'magenta': Qt.magenta,
                'cyan': Qt.cyan,
                'black': Qt.black,
                'white': Qt.white
            }
            return color_map.get(color_string, Qt.black)


# Classes
class Legend(QWidget):
    
    def __init__(self, color_manager):
        super().__init__()
        self.color_manager = color_manager
        self.radio_buttons = []  # Store references to radio buttons
        self.color_indicators = []  # Store references to color indicators
        # Group radio buttons to enforce exclusivity across containers
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        layout = QGridLayout()
        
        counter = 0
        for i, label in enumerate(config['labels']):
            if label['active']:
                # Create horizontal layout for radio button and color indicator
                item_layout = QHBoxLayout()
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(5)
                
                # Create radio button
                radiobutton = QRadioButton(label['name'])
                radiobutton.setFixedSize(QSize(64, 30))
                radiobutton.id = i
                radiobutton.name = label['name']
                if i == config['active_label']:
                    radiobutton.setChecked(True)
                radiobutton.toggled.connect(self.onClicked)
                # Add to exclusive group
                self.button_group.addButton(radiobutton, i)
                
                # Create color indicator
                color_indicator = QLabel()
                color_indicator.setFixedSize(QSize(20, 20))
                color_indicator.setStyleSheet("border: 1px solid black;")
                
                # Get color for this label
                color_string = self.color_manager.get_color_for_label(i, label['name'])
                qt_color = self.color_manager.get_qt_color(color_string)
                
                # Convert QColor to CSS color string
                if hasattr(qt_color, 'name') and qt_color.name():
                    css_color = qt_color.name()
                elif hasattr(qt_color, 'getRgb'):
                    # Fallback: convert RGB values to hex
                    r, g, b, a = qt_color.getRgb()
                    css_color = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    # For GlobalColor objects, use a mapping
                    color_mapping = {
                        Qt.red: '#ff0000',
                        Qt.blue: '#0000ff', 
                        Qt.green: '#008000',
                        Qt.yellow: '#ffff00',
                        Qt.magenta: '#ff00ff',
                        Qt.cyan: '#00ffff',
                        Qt.black: '#000000',
                        Qt.white: '#ffffff'
                    }
                    css_color = color_mapping.get(qt_color, '#000000')
                
                color_indicator.setStyleSheet(f"background-color: {css_color}; border: 1px solid black;")
                
                # Add to layout
                item_layout.addWidget(radiobutton)
                item_layout.addWidget(color_indicator)
                
                # Create container widget
                container = QWidget()
                container.setLayout(item_layout)
                container.setFixedSize(QSize(90, 30))
                
                layout.addWidget(container, counter % 2, counter // 2)
                self.radio_buttons.append(radiobutton)  # Store reference
                self.color_indicators.append(color_indicator)  # Store reference
                counter += 1
        
        self.setLayout(layout)


    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            config['active_label'] = radioButton.id
            print(f"{radioButton.name} toggled!")


class Label(QWidget):

    def __init__(self, id, color_manager):
        super().__init__()
        self.id = id
        self.color_manager = color_manager
        self.name = config['labels'][self.id]['name']
        self.active = config['labels'][self.id]['active']
        
        # Get or assign color dynamically
        self.color = self.color_manager.get_color_for_label(self.id, self.name)

        self.qlabel = QLabel(str(id))
        self.qlabel.setAlignment(Qt.AlignCenter)
        self.textbox = QLineEdit()
        self.textbox.setFixedSize(QSize(128, 24))
        self.textbox.setStyleSheet(f"QLineEdit{{background : {self.color}}}")
        self.textbox.setText(self.name)
        self.textbox.textChanged.connect(self.update_text)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet(
            "QCheckBox::indicator{width: 24px; height: 24px;}")
        self.checkbox.stateChanged.connect(self.update_status)
        self.checkbox.setChecked(self.active)
        self.update_status()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.qlabel)
        layout.addWidget(self.textbox)
        layout.addWidget(self.checkbox)

        self.setLayout(layout)
        
    def update_status(self):
        self.textbox.setEnabled(self.checkbox.isChecked())
        if self.checkbox.isChecked():
            config['labels'][self.id]['active'] = True
        else:
            config['labels'][self.id]['active'] = False

    def update_text(self):
        config['labels'][self.id]['name'] = self.textbox.text()


class TextBox(QWidget):

    def __init__(self, key, title, default):
        super().__init__()
        
        self.key = key
        self.title = QLabel(title)
        self.title.setFixedHeight(32)
        
        self.textbox = QLineEdit()
        self.textbox.setFixedSize(64, 32)
        self.textbox.setText(str(default))
        self.textbox.textChanged.connect(self.update_text)

        layout = QHBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.textbox)

        self.setLayout(layout)

    def update_text(self):
        if self.textbox.text() != '':
            if isinstance(config[self.key], str):
                config[self.key] = self.textbox.text()
            elif isinstance(config[self.key], int):
                config[self.key] = int(self.textbox.text())


class SettingWindow(QWidget):

    def __init__(self, color_manager, *args, **kwargs):
        super().__init__()
        self.color_manager = color_manager
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.dialog = QFileDialog()
        self.dialog.setFileMode(QFileDialog.Directory)
        
        self.title_1 = QLabel("labels")
        self.title_1.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_1)
        self.labels = [Label(id, color_manager) for id in range(len(config['labels']))]
        for i in range(len(config['labels'])):
            layout.addWidget(self.labels[i])
        
        self.title_2 = QLabel("configuration")
        self.title_2.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_2)
        
        self.choosebutton = QPushButton("select output directory")
        self.choosebutton.pressed.connect(self.choose_output_dir)
        self.choosebutton.setFixedHeight(32)
        layout.addWidget(self.choosebutton)
        
        # Color scheme selection removed; default scheme is always used
        
        self.image_key = TextBox(
            'image_key', 'image key: ', config['image_key'])
        self.data_key = TextBox(
            'data_key', 'data key: ', config['data_key'])
        self.tile_size = TextBox(
            'tile_size', 'tile size [pixels]', config['tile_size'])
        self.x_size = TextBox(
            'x_size', 'horizontal tile count', config['x_size'])
        self.y_size = TextBox(
            'y_size', 'vertical tile count', config['y_size'])

        layout.addWidget(self.image_key)
        layout.addWidget(self.data_key)
        layout.addWidget(self.tile_size)
        layout.addWidget(self.x_size)
        layout.addWidget(self.y_size)
        
        self.setLayout(layout)

    def choose_output_dir(self):
        config['output_dir'] = self.dialog.getExistingDirectory(
            self.choosebutton, "Open Directory", '',
            QFileDialog.ShowDirsOnly)

    # Color scheme changes are disabled; default is always used


class Pos(QWidget):
    
    def __init__(self, id, qImage, label, *args, **kwargs):
        super(Pos, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(config['tile_size'], config['tile_size']))
        self.id = id
        self.image = qImage
        self.label = label
        
    def reset(self, id, qImage, label):
        self.id = id
        self.image = qImage
        self.label = label
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        r = event.rect()
        p.drawImage(r, self.image)
        # p.drawPixmap(r, QPixmap(self.image))
        color = self.get_color()
        pen = QPen(color)
        pen.setWidth(4)
        p.setPen(pen)
        p.drawRect(r)
        
    def flag(self):
        self.label = config['active_label']
        logger.info(f"Event {self.id} is selected!")
        self.update()
        
        # Update all channel views of the same image
        self.update_all_channels_for_image()
        
    def update_all_channels_for_image(self):
        """Update all channel views of the same image with the current label."""
        # Find the parent container that holds all channels for this image
        parent = self.parent()
        if parent and hasattr(parent, 'layout'):
            layout = parent.layout()
            if layout:
                # Update all widgets in the same layout (all channels of the same image)
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        widget = item.widget()
                        if hasattr(widget, 'id') and widget.id == self.id:
                            widget.label = self.label
                            widget.update()

    def junk(self):
        self.label = 0
        logger.info(f"Event {self.id} is discarded!")
        self.update()
        
        # Update all channel views of the same image
        self.update_all_channels_for_image()

    def get_color(self):
        # Resolve label name
        try:
            label_name = config['labels'][self.label]['name']
        except Exception:
            label_name = f"label_{self.label}"

        # Prefer ColorManager for dynamic color resolution
        if hasattr(self, 'color_manager') and self.color_manager:
            color_string = self.color_manager.get_color_for_label(self.label, label_name)
            return self.color_manager.get_qt_color(color_string)

        # Fallback to a deterministic basic palette
        fallback_colors = [Qt.red, Qt.blue, Qt.green, Qt.yellow, Qt.magenta, Qt.cyan, Qt.black]
        return fallback_colors[self.label % len(fallback_colors)]

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.junk()
        elif event.button() == Qt.LeftButton:
            self.flag()


class MainWindow(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        #self.setStyleSheet("background-color: black;")
        self.current_page = 0
        self.n_pages = 0
        self.f_name = 'Empty'
        
        # Initialize managers
        self.image_cache = None
        self.color_manager = ColorManager()
        
        # Channel selection
        self.selected_channels = ['composite']
        self.channel_widgets = []
        
        # Color scheme selection disabled; always default
        
        self.open_settings()

        self.dialog = QFileDialog()
        self.dialog.setFileMode(QFileDialog.AnyFile)
        
        self.selectallbutton = QToolButton()
        self.selectallbutton.setText("All")
        self.selectallbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.selectallbutton.setFixedSize(QSize(64, 64))
        self.selectallbutton.setIconSize(QSize(32, 32))
        self.selectallbutton.setIcon(QIcon("./icons/check_all.png"))
        self.selectallbutton.pressed.connect(self.selectAll)
        
        self.selectnonebutton = QToolButton()
        self.selectnonebutton.setText("None")
        self.selectnonebutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.selectnonebutton.setFixedSize(QSize(64, 64))
        self.selectnonebutton.setIconSize(QSize(32, 32))
        self.selectnonebutton.setIcon(QIcon("./icons/uncheck_all.png"))
        self.selectnonebutton.pressed.connect(self.selectNone)
        
        self.nextbutton = QToolButton()
        self.nextbutton.setText("Next")
        self.nextbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.nextbutton.setFixedSize(QSize(64, 64))
        self.nextbutton.setIconSize(QSize(32, 32))
        self.nextbutton.setIcon(QIcon("./icons/Right.png"))
        self.nextbutton.pressed.connect(self.nextPage)
        
        self.prevbutton = QToolButton()
        self.prevbutton.setText("Prev")
        self.prevbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.prevbutton.setFixedSize(QSize(64, 64))
        self.prevbutton.setIconSize(QSize(32, 32))
        self.prevbutton.setIcon(QIcon("./icons/Left.png"))
        self.prevbutton.pressed.connect(self.prevPage)
         
        self.savebutton = QToolButton()
        self.savebutton.setText("Save")
        self.savebutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.savebutton.setFixedSize(QSize(64, 64))
        self.savebutton.setIconSize(QSize(32, 32))
        self.savebutton.setIcon(QIcon("./icons/Save.png"))
        self.savebutton.pressed.connect(self.save_data)
        
        self.loadbutton = QToolButton()
        self.loadbutton.setText("Load")
        self.loadbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.loadbutton.setIconSize(QSize(32, 32))
        self.loadbutton.setIcon(QIcon("./icons/Open.png"))
        self.loadbutton.setFixedSize(QSize(64, 64))
        self.loadbutton.pressed.connect(self.load_data)
       
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        self.grid.setContentsMargins(0, 0, 0, 0)
        # Wrap grid in a QWidget so we can control and measure its size
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(self.grid)
        self.grid_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.page_number = QLabel()
        self.page_number.setFixedSize(QSize(64, 64))
        self.page_number.setAlignment(Qt.AlignCenter)
        self.page_number.setText(f"{self.f_name}\n\n"
                                 f"{self.current_page} / {self.n_pages}")
        self.legend = Legend(self.color_manager)
        
        # Channel selection controls
        self.create_channel_controls()
        
        # Grid size controls
        self.create_grid_controls()

        # Control panels (wrap in a QWidget so we can get sizeHint reliably)
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setContentsMargins(5, 5, 5, 5)
        control_panel_layout.addWidget(self.channel_group)
        control_panel_layout.addWidget(self.grid_group)
        
        key_box = QHBoxLayout()
        key_box.setContentsMargins(0, 0, 0, 0)
        key_box.addWidget(self.legend)
        key_box.addWidget(self.selectallbutton)
        key_box.addWidget(self.selectnonebutton)
        #key_box.addWidget(self.settingbutton)
        key_box.addWidget(self.prevbutton)
        key_box.addWidget(self.page_number)
        key_box.addWidget(self.nextbutton)
        key_box.addWidget(self.savebutton)
        key_box.addWidget(self.loadbutton)
        
        control_panel_layout.addLayout(key_box)
        self.control_panel_widget = QWidget()
        self.control_panel_widget.setLayout(control_panel_layout)
               
        main_box = QVBoxLayout()
        main_box.setContentsMargins(0, 0, 0, 0)
        main_box.setSpacing(0)
        main_box.addWidget(self.grid_widget)
        main_box.addWidget(self.control_panel_widget)

        main_widget = QWidget()
        main_widget.setLayout(main_box)
        self.setCentralWidget(main_widget)
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
        self.load_data(init_map=True)
        self.show()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for common actions."""
        # Clear image cache shortcut (Ctrl+Shift+C)
        clear_cache_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        clear_cache_shortcut.activated.connect(self.clear_image_cache)
        
        # Page navigation shortcuts
        next_page_shortcut = QShortcut(QKeySequence("Right"), self)
        next_page_shortcut.activated.connect(self.nextPage)
        
        prev_page_shortcut = QShortcut(QKeySequence("Left"), self)
        prev_page_shortcut.activated.connect(self.prevPage)
        
        # Label selection shortcuts (number keys 0-9)
        for i in range(10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda checked=False, label_id=i: self.select_label(label_id))
        
        # Additional navigation shortcuts
        up_shortcut = QShortcut(QKeySequence("Up"), self)
        up_shortcut.activated.connect(self.prevPage)
        
        down_shortcut = QShortcut(QKeySequence("Down"), self)
        down_shortcut.activated.connect(self.nextPage)
        
        # Help shortcut (F1)
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_help)

    def select_label(self, label_id):
        """Select a label using keyboard shortcut."""
        # Check if the label exists and is active
        if label_id < len(config['labels']) and config['labels'][label_id]['active']:
            config['active_label'] = label_id
            logger.info(f"Selected label: {config['labels'][label_id]['name']} (ID: {label_id})")
            
            # Update the legend to reflect the selection
            if hasattr(self, 'legend') and self.legend:
                self.update_legend_selection()
        else:
            logger.warning(f"Label {label_id} is not available or not active")

    def update_legend_selection(self):
        """Update the legend to show the currently selected label."""
        if hasattr(self, 'legend') and self.legend and hasattr(self.legend, 'radio_buttons'):
            # Update all radio buttons
            for radio_button in self.legend.radio_buttons:
                if hasattr(radio_button, 'id'):
                    radio_button.setChecked(radio_button.id == config['active_label'])
            
            # Don't update color indicators when just selecting a label
            # Color indicators should only change when color scheme changes

    def update_color_indicators(self):
        """Update the color indicators to reflect current colors."""
        if hasattr(self, 'legend') and self.legend and hasattr(self.legend, 'color_indicators'):
            for i, color_indicator in enumerate(self.legend.color_indicators):
                if i < len(config['labels']) and config['labels'][i]['active']:
                    # Get current color for this label
                    label_name = config['labels'][i]['name']
                    color_string = self.color_manager.get_color_for_label(i, label_name)
                    qt_color = self.color_manager.get_qt_color(color_string)
                    
                    # Convert QColor to CSS color string
                    if hasattr(qt_color, 'name') and qt_color.name():
                        css_color = qt_color.name()
                    elif hasattr(qt_color, 'getRgb'):
                        # Fallback: convert RGB values to hex
                        r, g, b, a = qt_color.getRgb()
                        css_color = f"#{r:02x}{g:02x}{b:02x}"
                    else:
                        # For GlobalColor objects, use a mapping
                        color_mapping = {
                            Qt.red: '#ff0000',
                            Qt.blue: '#0000ff', 
                            Qt.green: '#008000',
                            Qt.yellow: '#ffff00',
                            Qt.magenta: '#ff00ff',
                            Qt.cyan: '#00ffff',
                            Qt.black: '#000000',
                            Qt.white: '#ffffff'
                        }
                        css_color = color_mapping.get(qt_color, '#000000')
                    
                    color_indicator.setStyleSheet(f"background-color: {css_color}; border: 1px solid black;")

    def show_help(self):
        """Show keyboard shortcuts help dialog."""
        help_text = """
Keyboard Shortcuts:

Navigation:
• Left/Right Arrow Keys - Previous/Next page
• Up/Down Arrow Keys - Previous/Next page
• F1 - Show this help

Label Selection:
• 0-9 - Select label by number (if available and active)

Actions:
• Left Click - Select/flag an image tile
• Right Click - Mark an image tile as junk
• Ctrl+Shift+C - Clear image cache

Channel Selection:
• Use checkboxes in the Channels panel

Grid Size:
• Use X/Y spinboxes and Apply button
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def create_channel_controls(self):
        """Create channel selection controls."""
        self.channel_group = QGroupBox("Channels")
        self.channel_group.setFixedHeight(80)
        channel_layout = QHBoxLayout()
        
        # Channel checkboxes
        self.channel_checkboxes = {}
        channels = ['composite', '0', '1', '2', '3']
        
        for i, channel in enumerate(channels):
            checkbox = QCheckBox(f"Ch {channel}")
            checkbox.setChecked(channel == 'composite')  # Default to composite
            checkbox.stateChanged.connect(self.on_channel_changed)
            self.channel_checkboxes[channel] = checkbox
            channel_layout.addWidget(checkbox)
        
        self.channel_group.setLayout(channel_layout)

    def create_grid_controls(self):
        """Create grid size controls."""
        self.grid_group = QGroupBox("Grid Size")
        self.grid_group.setFixedHeight(80)
        grid_layout = QHBoxLayout()
        
        # X size control
        x_label = QLabel("X:")
        x_label.setFixedWidth(20)
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(1, 50)
        self.x_spinbox.setValue(config.get('x_size', 15))
        self.x_spinbox.valueChanged.connect(self.on_grid_size_changed)
        
        # Y size control
        y_label = QLabel("Y:")
        y_label.setFixedWidth(20)
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(1, 50)
        self.y_spinbox.setValue(config.get('y_size', 15))
        self.y_spinbox.valueChanged.connect(self.on_grid_size_changed)
        
        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_grid_changes)
        
        grid_layout.addWidget(x_label)
        grid_layout.addWidget(self.x_spinbox)
        grid_layout.addWidget(y_label)
        grid_layout.addWidget(self.y_spinbox)
        grid_layout.addWidget(apply_button)
        
        self.grid_group.setLayout(grid_layout)

    def on_channel_changed(self):
        """Handle channel selection changes."""
        selected = []
        for channel, checkbox in self.channel_checkboxes.items():
            if checkbox.isChecked():
                selected.append(channel)
        
        if not selected:
            # If nothing selected, default to composite
            self.channel_checkboxes['composite'].setChecked(True)
            selected = ['composite']
        
        self.selected_channels = selected
        
        # Update image cache if it exists
        if self.image_cache:
            self.image_cache.set_selected_channels(selected)
        
        # Apply grid changes as if the Apply button was clicked
        self.apply_grid_changes()

    def on_grid_size_changed(self):
        """Handle grid size changes (just update config, don't apply yet)."""
        config['x_size'] = self.x_spinbox.value()
        config['y_size'] = self.y_spinbox.value()

    def apply_grid_changes(self):
        """Apply grid size changes and refresh display."""
        self.x_size = config['x_size']
        self.y_size = config['y_size']
        
        # Recalculate pages
        if hasattr(self, 'n_events') and self.n_events > 0:
            self.n_pages = 1 + self.n_events // (self.x_size * self.y_size)
            self.n_tiles = self.n_pages * (self.x_size * self.y_size)
            self.current_page = min(self.current_page, self.n_pages)
            self.update_page_number()
        
        # Clear and rebuild grid
        self.clear_grid()
        self.refresh_display()
        
        # Force immediate window resize
        self.resize_window_immediately()
        
        # Force a tight layout update
        self.force_tight_layout()

    def clear_grid(self):
        """Clear all widgets from the grid."""
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def resize_window_to_grid(self):
        """Resize window to accommodate the current grid size."""
        if not hasattr(self, 'x_size') or not hasattr(self, 'y_size'):
            return
            
        # Calculate required size dynamically using size hints
        tile_size = config.get('tile_size', 85)
        num_channels = len(self.selected_channels)

        grid_width = tile_size * self.x_size * num_channels
        grid_height = tile_size * self.y_size

        # Measure control panel height dynamically
        control_height = self.control_panel_widget.sizeHint().height() if hasattr(self, 'control_panel_widget') else 200
        margin_width = 20

        # Calculate final window size
        width = max(grid_width + margin_width, 300)
        height = max(grid_height + control_height, 250)

        # Resize without locking fixed size to allow further growth
        self.resize(width, height)

        # Process events to ensure resize happens immediately
        QApplication.processEvents()

        # Force the layout to be tight
        self.force_tight_layout()
        
        # Force the layout to be tight
        self.force_tight_layout()

    def resize_window_immediately(self):
        """Force immediate window resize with proper calculations."""
        if not hasattr(self, 'x_size') or not hasattr(self, 'y_size'):
            return
            
        # Calculate required size dynamically
        tile_size = config.get('tile_size', 85)
        num_channels = len(self.selected_channels)

        grid_width = tile_size * self.x_size * num_channels
        grid_height = tile_size * self.y_size

        control_height = self.control_panel_widget.sizeHint().height() if hasattr(self, 'control_panel_widget') else 200
        margin_width = 20

        width = max(grid_width + margin_width, 300)
        height = max(grid_height + control_height, 250)

        # Resize without locking; then process events
        self.resize(width, height)
        QApplication.processEvents()

        # Reset to allow future resizing
        self.setMinimumSize(300, 250)
        self.setMaximumSize(16777215, 16777215)  # Qt's maximum size

    def force_tight_layout(self):
        """Force the layout to be tight with no extra space."""
        if hasattr(self, 'grid') and self.grid is not None:
            # Update the grid layout to be tight
            self.grid.setSpacing(0)
            self.grid.setContentsMargins(0, 0, 0, 0)
            
            # Ensure grid_widget keeps the grid area large enough
            tile_size = config.get('tile_size', 85)
            num_channels = len(self.selected_channels)
            grid_width = tile_size * self.x_size * num_channels
            grid_height = tile_size * self.y_size
            if hasattr(self, 'grid_widget'):
                self.grid_widget.setMinimumSize(grid_width, grid_height)

            # Force update of all widgets
            self.update()
            if self.centralWidget():
                self.centralWidget().update()
                self.centralWidget().layout().activate()

    def setup_channel_controls(self):
        """Set up channel controls based on available channels."""
        if hasattr(self, 'n_channels') and self.n_channels > 0:
            # Enable/disable channel checkboxes based on available channels
            for channel, checkbox in self.channel_checkboxes.items():
                if channel == 'composite':
                    checkbox.setEnabled(True)
                else:
                    try:
                        channel_idx = int(channel)
                        checkbox.setEnabled(channel_idx < self.n_channels)
                    except ValueError:
                        checkbox.setEnabled(False)

    def calc_index(self, x, y):
        return((self.current_page - 1) * self.x_size * self.y_size \
               + x + self.x_size * y)
    
    def get_image(self, id, mode, channel_mode='composite'):
        if mode == 'rgb':
            if self.image_cache:
                # Use cached image
                image_data = self.image_cache.get_image(id, channel_mode)
                if image_data is not None:
                    # Ensure dimensions
                    h, w, c = image_data.shape
                    if (h != self.im_h) or (w != self.im_w) or (c != 3):
                        self.im_h, self.im_w = h, w
                    arr = np.ascontiguousarray(image_data)
                    qimg = QImage(arr.data, self.im_w, self.im_h, self.im_w * 3, QImage.Format_RGB888)
                    return qimg, arr
                else:
                    empty_image = np.zeros((self.im_h, self.im_w, 3), dtype=np.uint8)
                    arr = np.ascontiguousarray(empty_image)
                    qimg = QImage(arr.data, self.im_w, self.im_h, self.im_w * 3, QImage.Format_RGB888)
                    return qimg, arr
            else:
                # Fallback to global images array
                global images
                if id < len(images):
                    img = images[id]
                    if channel_mode == 'composite':
                        if img.dtype == np.uint16:
                            h, w, c = img.shape
                            img = channels2rgb8bit(img.reshape(1, h, w, c))[0]
                    else:
                        # Single channel mode
                        if img.ndim == 3:
                            try:
                                channel_idx = int(channel_mode)
                                if 0 <= channel_idx < img.shape[2]:
                                    single_channel = img[:, :, channel_idx]
                                    if single_channel.dtype == np.uint16:
                                        single_channel = (single_channel // 256).astype(np.uint8)
                                    img = np.stack([single_channel, single_channel, single_channel], axis=2)
                                else:
                                    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                            except (ValueError, IndexError):
                                img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    arr = np.ascontiguousarray(img)
                    qimg = QImage(arr.data, self.im_w, self.im_h, self.im_w * 3, QImage.Format_RGB888)
                    return qimg, arr
                else:
                    empty_image = np.zeros((self.im_h, self.im_w, 3), dtype=np.uint8)
                    arr = np.ascontiguousarray(empty_image)
                    qimg = QImage(arr.data, self.im_w, self.im_h, self.im_w * 3, QImage.Format_RGB888)
                    return qimg, arr

    def get_label(self, id):
        global df
        if id >= self.n_events:
            return 0
        else:
            return df.label.iat[id]

    def init_map(self):
        self.create_image_grid()

    def create_image_grid(self):
        """Create the image grid with selected channels."""
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
                id = self.calc_index(x, y)
                label = self.get_label(id)
                
                # Create a horizontal layout for multiple channels
                channel_layout = QHBoxLayout()
                channel_layout.setSpacing(0)
                channel_layout.setContentsMargins(0, 0, 0, 0)
                
                for channel in self.selected_channels:
                    qImage, arr = self.get_image(id, mode='rgb', channel_mode=channel)
                    w = Pos(id, qImage, label)
                    w.color_manager = self.color_manager
                    w._qimage_buffer = arr
                    w._channel = channel
                    channel_layout.addWidget(w)
                
                # Create container widget for this grid position
                container = QWidget()
                container.setLayout(channel_layout)
                container.setFixedSize(
                    len(self.selected_channels) * config['tile_size'], 
                    config['tile_size']
                )
                container.setContentsMargins(0, 0, 0, 0)
                container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                self.grid.addWidget(container, y, x)

        # After building the grid, enforce the grid_widget minimum size so tiles don't shrink
        tile_size = config.get('tile_size', 85)
        num_channels = len(self.selected_channels)
        grid_width = tile_size * self.x_size * num_channels
        grid_height = tile_size * self.y_size
        if hasattr(self, 'grid_widget'):
            self.grid_widget.setMinimumSize(grid_width, grid_height)

    def refresh_display(self):
        """Refresh the entire display."""
        if hasattr(self, 'grid') and self.grid is not None:
            self.clear_grid()
            self.create_image_grid()

    def reset_map(self):
        """Reset the map with current channel selection."""
        self.refresh_display()
    
    def update_page_number(self):
        self.page_number.setText(f"{self.f_name}\n\n"
                                 f"{self.current_page} / {self.n_pages}")

    def nextPage(self):
        if self.current_page < self.n_pages:
            self.current_page += 1
            self.update_page_number()
            logger.info(f"Page: {self.current_page}")
            
            # Preload next page images
            if self.image_cache:
                start_id = (self.current_page - 1) * self.x_size * self.y_size
                end_id = min(start_id + self.x_size * self.y_size, self.n_events)
                self.image_cache.preload_range(start_id, end_id)
        else:
            logger.warning("This is the last page!")
        self.save_labels()
        self.reset_map()
        
    def prevPage(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_page_number()
            logger.info(f"Page: {self.current_page}")
            
            # Preload previous page images
            if self.image_cache:
                start_id = (self.current_page - 1) * self.x_size * self.y_size
                end_id = min(start_id + self.x_size * self.y_size, self.n_events)
                self.image_cache.preload_range(start_id, end_id)
        else:
            logger.warning("This is the first page!")
        self.save_labels()
        self.reset_map()
        
    def selectAll(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                container = self.grid.itemAtPosition(y, x).widget()
                if container:
                    # Apply to all Pos widgets in the container
                    channel_layout = container.layout()
                    if channel_layout:
                        for i in range(channel_layout.count()):
                            w = channel_layout.itemAt(i).widget()
                            if hasattr(w, 'flag'):
                                w.flag()
                
    def selectNone(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                container = self.grid.itemAtPosition(y, x).widget()
                if container:
                    # Apply to all Pos widgets in the container
                    channel_layout = container.layout()
                    if channel_layout:
                        for i in range(channel_layout.count()):
                            w = channel_layout.itemAt(i).widget()
                            if hasattr(w, 'junk'):
                                w.junk()
                
    def save_labels(self):
        global df
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                container = self.grid.itemAtPosition(y, x).widget()
                if container:
                    # Get the first Pos widget from the container (they all have the same id and label)
                    channel_layout = container.layout()
                    if channel_layout and channel_layout.count() > 0:
                        w = channel_layout.itemAt(0).widget()
                        if hasattr(w, 'id') and w.id < self.n_events:
                            df.label.iat[w.id] = w.label
                
        logger.info(f"Selection: {sum(df.label)}")

    def open_settings(self):
        main_dialog = QDialog()
        layout = QVBoxLayout()
        setting_window = SettingWindow(self.color_manager)
        layout.addWidget(setting_window)
        main_dialog.setLayout(layout)
        main_dialog.setWindowTitle('Settings')
        main_dialog.setWindowModality(Qt.ApplicationModal)
        main_dialog.exec_()
        save_config()
        self.deploy_config()

    def deploy_config(self):
        self.x_size = config['x_size']
        self.y_size = config['y_size']
        
        # Color scheme is fixed to default; no updates needed

    def refresh_label_colors(self):
        """Refresh colors for all visible tiles when color scheme changes."""
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                container = self.grid.itemAtPosition(y, x).widget()
                if container:
                    # Update all Pos widgets in the container
                    channel_layout = container.layout()
                    if channel_layout:
                        for i in range(channel_layout.count()):
                            w = channel_layout.itemAt(i).widget()
                            if w:
                                w.update()  # This will trigger a repaint with new colors
        
        # Update legend color indicators
        if hasattr(self, 'legend') and self.legend and hasattr(self.legend, 'color_indicators'):
            self.update_color_indicators()

    def clear_image_cache(self):
        """Clear the image cache to free memory."""
        if self.image_cache:
            self.image_cache.clear_cache()
            logger.info("Image cache cleared to free memory")

    def load_data(self, init_map=False):
        global images
        global df

        self.f_path, _ = self.dialog.getOpenFileName(
            self.loadbutton, "Open File", '', "HDF files (*.hdf5)")

        if not self.f_path:
            return
        try:
            self.f_name = os.path.basename(self.f_path).replace('.hdf5', '')
            logger.info(f"loading input data from: {self.f_path}")

            # Initialize image cache manager
            cache_size = config.get('image_cache_size', 100)
            self.image_cache = ImageCacheManager(self.f_path, config['image_key'], cache_size=cache_size)
            
            # Load data (not images - they'll be loaded dynamically)
            with h5py.File(self.f_path, 'r') as file:
                self.input_keys = list(file.keys())
                logger.debug(f"Input file keys: {self.input_keys}")
                
                # Get image shape without loading all images
                if config['image_key'] in self.input_keys:
                    with h5py.File(self.f_path, 'r') as f:
                        image_dataset = f[config['image_key']]
                        self.im_shape = image_dataset.shape
                        logger.info(f"Image dataset shape: {self.im_shape}")
                else:
                    logger.error("Images not found in input file!")
                    sys.exit(-1)

            if config['data_key'] in self.input_keys:
                df = pd.read_hdf(self.f_path, config['data_key'])
                logger.info(f"Loaded data with size: {df.shape}")
                logger.debug(f"Types of data columns:\n{df.dtypes}")
            else:
                logger.info(f"Data not found in input file!")
                sys.exit(-1)

        except Exception as e:
            QMessageBox.warning(
                self, 'Error', f"The following error occured:\n{type(e)}: {e}")

        self.n_events   = self.im_shape[0]
        self.im_h       = self.im_shape[1]
        self.im_w       = self.im_shape[2]
        self.n_channels = self.im_shape[3]
        self.n_pages    = 1 + self.n_events // (self.x_size * self.y_size)
        self.n_tiles    = self.n_pages * (self.x_size * self.y_size)

        # Ensure we start from page 1 before any preloading
        self.current_page = 1

        # Set up channel controls based on available channels
        self.setup_channel_controls()
        
        # Preload first page of images for better performance
        if self.image_cache:
            start_id = (self.current_page - 1) * self.x_size * self.y_size
            end_id = min(start_id + self.x_size * self.y_size, self.n_events)
            self.image_cache.preload_range(start_id, end_id)

        if 'label' not in df.columns:
            df['label'] = np.zeros(self.n_events, dtype='uint8')

        self.update_page_number()
        if init_map:
            self.init_map()
        else:
            self.reset_map()

    def save_data(self, export_txt=True):
        global df
        self.save_labels()
        
        # Close any open file handles before saving
        if hasattr(self, 'image_cache') and self.image_cache is not None:
            self.image_cache.close_file()
        
        # saving annotations to hdf5 file
        try:
            df.to_hdf(self.f_path, key=config['data_key'], mode='r+')
            # saving label keymap
            with h5py.File(self.f_path, 'r+') as file:
                if 'labels' in file.keys():
                    del file['labels']
                file.create_dataset(
                    'labels', data=[item['name'] for item in config['labels']])
            logger.info("Stored data in HDF file!")
        except Exception as e:
            logger.error(f"Error saving data to HDF5 file: {e}")
            raise
        finally:
            # Reopen the image cache after saving
            if hasattr(self, 'image_cache') and self.image_cache is not None:
                self.image_cache.open_file()
        # exporting data to a txt file if requested
        if export_txt:
            export_path = f"{config['output_dir']}/{self.f_name}.txt"
            df.to_csv(export_path, index=False, sep='\t')
            logger.info(f"Exported data to {export_path}")

    def closeEvent(self,event):
        result = QMessageBox.question(self,
                      "Confirm Exit...",
                      "Are you sure you want to exit?",
                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes:
            # Clean up image cache
            if self.image_cache:
                self.image_cache.close_file()
            event.accept()

# Functions
def load_config():
    global config
    if not os.path.exists(config_path):
        sys.exit("config file does not exist!")
    else:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        config['tile_size'] = 2 * (config['tile_size'] // 2) + 1

def save_config():
    global config
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    


def main():
    load_config()
    app = QApplication([])
    window = MainWindow()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == '__main__':
    main()
    
# Add input key-ID window

## Next Versions:

# Change it to dark theme
# Scale images
# Improve images
# Add option 3-color or gray-scale
# Add multiple selection by dragging mouse click
# Filter using size
# Show event data while hovering cursor over the item and waiting


##### To be used later
        #self.datacombo = QComboBox()
        #self.datacombo.setFixedSize(QSize(128, 30))
        #self.datacombo.addItems([str(i) for i in range(10)])
        #self.datacombo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        #self.load_box = QVBoxLayout()
        #self.load_box.addWidget(self.loadbutton)
        #self.load_box.addWidget(self.datacombo)

##### To be used later
#    def closeEvent(self,event):
#        result = QMessageBox.question(self,
#                      "Confirm Exit...",
#                      "Are you sure you want to exit ?",
#                      QMessageBox.Yes| QMessageBox.No)
#        event.ignore()
#
#        if result == QMessageBox.Yes:
#            event.accept()

##### For future use
        #self.settingbutton = QToolButton()
        #self.settingbutton.setText("Setting")
        #self.settingbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        #self.settingbutton.setFixedSize(QSize(64, 64))
        #self.settingbutton.setIconSize(QSize(32, 32))
        #self.settingbutton.setIcon(QIcon("./icons/Settings.png"))
        #self.settingbutton.clicked.connect(self.open_settings)
        
