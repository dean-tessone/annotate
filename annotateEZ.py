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
        if(image.shape[3] == 4):
            image = image[:, :, :, 0:3] + np.expand_dims(image[:, :, :, 3], 3)
        
    elif(len(image.shape) == 3):
        image[:, :, 0:3] = image[:, :, [1, 2, 0]]
        if(image.shape[2] == 4):
            image = image[:, :, 0:3] + np.expand_dims(image[:, :, 3], 2)
        
    image[image > 65535] = 65535
    image = (image / 256).astype('uint8')
    return(image)


# Classes
class Label(QWidget):

    def __init__(self, id):
        super().__init__()
        self.id = id
        self.color = config['labels'][self.id]['color']
        self.name = config['labels'][self.id]['name']
        self.active = config['labels'][self.id]['active']

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

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.dialog = QFileDialog()
        self.dialog.setFileMode(QFileDialog.Directory)
        
        self.title_1 = QLabel("labels")
        self.title_1.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_1)
        self.labels = [Label(id) for id in range(len(config['labels']))]
        for i in range(len(config['labels'])):
            layout.addWidget(self.labels[i])
        
        self.title_2 = QLabel("configuration")
        self.title_2.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_2)
        
        self.choosebutton = QPushButton("select output directory")
        self.choosebutton.pressed.connect(self.choose_output_dir)
        self.choosebutton.setFixedHeight(32)
        layout.addWidget(self.choosebutton)
        
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
        if self.label == config['active_label']:
            color = Qt.red
        else:
            color = Qt.black
        pen = QPen(color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r)
        
    def flag(self):
        self.label = config['active_label']
        logger.info(f"Event {self.id} is selected!")
        self.update()
        
    def junk(self):
        self.label = 0
        logger.info(f"Event {self.id} is discarded!")
        self.update()
    
    def get_label(self):
        return(self.label)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.junk()
        elif event.button() == Qt.LeftButton:
            self.flag()


class MainWindow(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.current_page = 0
        self.n_pages = 0
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
        self.grid.setSpacing(1)
        
        
        self.page_number = QLabel()
        self.page_number.setFixedSize(QSize(64, 64))
        self.page_number.setAlignment(Qt.AlignCenter)
        self.page_number.setText(f"{self.current_page} / {self.n_pages}")

        key_box = QHBoxLayout()
        key_box.addWidget(self.selectallbutton)
        key_box.addWidget(self.selectnonebutton)
        #key_box.addWidget(self.settingbutton)
        key_box.addWidget(self.prevbutton)
        key_box.addWidget(self.page_number)
        key_box.addWidget(self.nextbutton)
        key_box.addWidget(self.savebutton)
        key_box.addWidget(self.loadbutton)
               
        main_box = QVBoxLayout()
        main_box.addLayout(self.grid)
        main_box.addLayout(key_box)

        main_widget = QWidget()
        main_widget.setLayout(main_box)
        self.setCentralWidget(main_widget)
        
        self.load_data(init_map=True)
        self.show()
    

    def calc_index(self, x, y):
        return((self.current_page - 1) * self.x_size * self.y_size \
               + x + self.x_size * y)
    
    def get_image(self, id, mode):
        global images
        if mode == 'rgb':
            return(
                QImage(
                    images[id].data, self.im_w, self.im_h,
                    self.im_w * 3, QImage.Format_RGB888)
            )

    def get_label(self, id):
        global df
        if id >= self.n_events:
            return None
        else:
            return df.label.iat[id]

    def init_map(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                id = self.calc_index(x, y)
                qImage = self.get_image(id, mode='rgb')
                label = self.get_label(id)
                w = Pos(id, qImage, label)
                self.grid.addWidget(w, y, x)

    def reset_map(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                id = self.calc_index(x, y)
                qImage = self.get_image(id, mode='rgb')
                label = self.get_label(id)
                w = self.grid.itemAtPosition(y, x).widget()
                w.reset(id, qImage, label)
    
    def update_page_number(self):
        self.page_number.setText(f"{self.current_page} / {self.n_pages}")

    def nextPage(self):
        if self.current_page < self.n_pages:
            self.current_page += 1
            self.update_page_number()
            logger.info(f"Page: {self.current_page}")
        else:
            logger.warning("This is the last page!")
        self.save_labels()
        self.reset_map()
        
    def prevPage(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_page_number()
            logger.info(f"Page: {self.current_page}")
        else:
            logger.warning("This is the first page!")
        self.save_labels()
        self.reset_map()
        
    def selectAll(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.flag()
                w.update()
                
    def selectNone(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.junk()
                w.update()
                
    def save_labels(self):
        global df
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                w = self.grid.itemAtPosition(y, x).widget()
                if w.id < self.n_events:
                    df.label.iat[w.id] = w.get_label()
                
        logger.info(f"Selection: {sum(df.label)}")

    def open_settings(self):
        main_dialog = QDialog()
        layout = QVBoxLayout()
        setting_window = SettingWindow()
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
        self.active_label = config['active_label']

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

            # Load images
            with h5py.File(self.f_path, 'r') as file:
                self.input_keys = list(file.keys())
                logger.debug(f"Input file keys: {self.input_keys}")
                if config['image_key'] in self.input_keys:
                    images = file.get(config['image_key'])[:]
                    logger.info(f"Loaded images with size : {images.shape}")
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

        self.im_shape   = images.shape
        self.n_events   = self.im_shape[0]
        self.im_h       = self.im_shape[1]
        self.im_w       = self.im_shape[2]
        self.n_channels = self.im_shape[3]
        self.n_pages    = 1 + self.n_events // (self.x_size * self.y_size)
        self.n_tiles    = self.n_pages * (self.x_size * self.y_size)

        if self.n_tiles > self.n_events:
            images = np.concatenate((images,
                np.zeros(
                    (self.n_tiles - self.n_events, self.im_h, self.im_w,
                    self.n_channels),
                    dtype=images.dtype)),
                axis=0
            )

        images = channels2rgb8bit(images)

        if 'label' not in df.columns:
            df['label'] = np.zeros(self.n_events, dtype='uint8')

        self.current_page = 1
        self.update_page_number()
        if init_map:
            self.init_map()
        else:
            self.reset_map()

    def save_data(self, export_txt=True):
        global df
        self.save_labels()
        # saving annotations to hdf5 file
        df.to_hdf(self.f_path, key=config['data_key'], mode='r+')
        # saving label keymap
        with h5py.File(self.f_path, 'r+') as file:
            if 'labels' in file.keys():
                del file['labels']
            file.create_dataset(
                'labels', data=[item['name'] for item in config['labels']])
        logger.info("Stored data in HDF file!")
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

##### To be used later
#class Legend(QWidget):
#    
#    def __init__(self):
#        super().__init__()
#        layout = QGridLayout()
#        self.setLayout(layout)
#        
#        counter = 0
#        for i, label in enumerate(config['labels']):
#            if label['active']:
#                radiobutton = QRadioButton(label['name'])
#                radiobutton.id = i
#                radiobutton.name = label['name']
#                radiobutton.toggled.connect(self.onClicked)
#                layout.addWidget(radiobutton, counter % 2, counter // 2)
#                counter += 1
#
#    def onClicked(self):
#        radioButton = self.sender()
#        if radioButton.isChecked():
#            print(radioButton.id, radioButton.name)


##### For future use
        #self.settingbutton = QToolButton()
        #self.settingbutton.setText("Setting")
        #self.settingbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        #self.settingbutton.setFixedSize(QSize(64, 64))
        #self.settingbutton.setIconSize(QSize(32, 32))
        #self.settingbutton.setIcon(QIcon("./icons/Settings.png"))
        #self.settingbutton.clicked.connect(self.open_settings)
        
