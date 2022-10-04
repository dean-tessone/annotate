from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
import sys
import os
import h5py
import yaml

# Input
config = {}
images = []
df = pd.DataFrame()

# Functions
def channels2rgb8bit(image):
    "Convert 4 channel images to 8-bit RGB color images."
    assert(image.dtype == 'uint16')
    image = image.astype('float')
    if(len(image.shape) == 4):
        image[:,:,:,0:3] = image[:,:,:,[1,2,0]]
        if(image.shape[3] == 4):
            image = image[:,:,:,0:3] + np.expand_dims(image[:,:,:,3], 3)
        
    elif(len(image.shape) == 3):
        image[:,:,0:3] = image[:,:,[1,2,0]]
        if(image.shape[2] == 4):
            image = image[:,:,0:3] + np.expand_dims(image[:,:,3], 2)
        
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
        self.textbox.setFixedSize(QSize(100, 24))
        self.textbox.setStyleSheet(f"QLineEdit{{background : {self.color}}}")
        self.textbox.setText(self.name)
        self.textbox.textChanged.connect(self.update_text)
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet(
            "QCheckBox::indicator{width: 24px; height: 24px;}")
        self.checkbox.stateChanged.connect(self.update_status)
        self.checkbox.setChecked(self.active)
        self.update_status()

        self.label_box = QHBoxLayout()
        self.label_box.setContentsMargins(0, 0, 0, 0)
        self.label_box.setSpacing(5)
        self.label_box.addWidget(self.qlabel)
        self.label_box.addWidget(self.textbox)
        self.label_box.addWidget(self.checkbox)

        self.label_widget = QWidget()
        self.label_widget.setLayout(self.label_box)
        
    def getLabelWidget(self):
        return(self.label_widget)

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

        self.box = QHBoxLayout()
        self.box.addWidget(self.title)
        self.box.addWidget(self.textbox)

        self.widget = QWidget()
        self.widget.setLayout(self.box)

    def get_widget(self):
        return(self.widget)

    def update_text(self):
        if isinstance(config[self.key], str):
            config[self.key] = self.textbox.text()
        elif isinstance(config[self.key], int):
            config[self.key] = int(self.textbox.text())




class SettingWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__()
        print(config['x_size'])
        self.dialog = QFileDialog()
        self.dialog.setFileMode(QFileDialog.Directory)
        
        # box 1: labels
        setting_box_1 = QVBoxLayout()
        setting_box_1.setContentsMargins(10, 10, 10, 10)
        setting_box_1.setSpacing(5)
        
        box_1_title = QLabel("labels")
        box_1_title.setAlignment(Qt.AlignCenter)
        setting_box_1.addWidget(box_1_title)
        self.labels = [Label(id) for id in range(len(config['labels']))]
        for i in range(len(config['labels'])):
            setting_box_1.addWidget(self.labels[i].getLabelWidget())

        setting_widget_1 = QWidget()
        setting_widget_1.setLayout(setting_box_1)
        
        # box 2: configuration
        box_2_title = QLabel("configuration")
        box_2_title.setAlignment(Qt.AlignCenter)
        self.choosebutton = QPushButton("select output directory")
        self.choosebutton.setFixedHeight(32)
        self.choosebutton.pressed.connect(self.choose_output_dir)
        
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

        setting_box_2 = QVBoxLayout()
        setting_box_2.addWidget(box_2_title)
        setting_box_2.addWidget(self.image_key.get_widget())
        setting_box_2.addWidget(self.data_key.get_widget())
        setting_box_2.addWidget(self.tile_size.get_widget())
        setting_box_2.addWidget(self.x_size.get_widget())
        setting_box_2.addWidget(self.y_size.get_widget())
        setting_box_2.addWidget(self.choosebutton)
        
        setting_widget_2 = QWidget()
        setting_widget_2.setLayout(setting_box_2)
        
        setting_box = QHBoxLayout()
        setting_box.addWidget(setting_widget_1)
        setting_box.addWidget(setting_widget_2)

        setting_widget = QWidget()
        setting_widget.setLayout(setting_box)
        self.setCentralWidget(setting_widget)

    def choose_output_dir(self):
        config['output_dir'] = self.dialog.getExistingDirectory(
            self.choosebutton, "Open Directory", '',
            QFileDialog.ShowDirsOnly)


class Legend(QWidget):
    
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        self.setLayout(layout)
        
        counter = 0
        for i, label in enumerate(config['labels']):
            if label['active']:
                radiobutton = QRadioButton(label['name'])
                radiobutton.id = i
                radiobutton.name = label['name']
                radiobutton.toggled.connect(self.onClicked)
                layout.addWidget(radiobutton, counter % 2, counter // 2)
                counter += 1

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print(radioButton.id, radioButton.name)


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
        if self.label:
            color = Qt.red
        else:
            color = Qt.black
        pen = QPen(color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r)
        
    def flag(self):
        self.label = True
        print(f"Event {self.id} is selected!")
        self.update()
        
    def junk(self):
        self.label = False
        print(f"Event {self.id} is discarded!")
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
        self.x_size = config['x_size']
        self.y_size = config['y_size']
        self.current_page = 1
        self.n_pages = 0
        self.active_label = 1
        
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
        
        self.settingbutton = QToolButton()
        self.settingbutton.setText("Setting")
        self.settingbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.settingbutton.setFixedSize(QSize(64, 64))
        self.settingbutton.setIconSize(QSize(32, 32))
        self.settingbutton.setIcon(QIcon("./icons/Settings.png"))
        self.settingbutton.clicked.connect(self.openSettings)
        
        self.nextbutton = QToolButton()
        self.nextbutton.setText("Next")
        self.nextbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.nextbutton.setFixedHeight(64)
        self.nextbutton.setIconSize(QSize(32, 32))
        self.nextbutton.setIcon(QIcon("./icons/Right.png"))
        self.nextbutton.pressed.connect(self.nextPage)
        
        self.prevbutton = QToolButton()
        self.prevbutton.setText("Prev")
        self.prevbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.prevbutton.setFixedHeight(64)
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
       
        self.load_data()
        
        self.page_number = QLabel()
        self.page_number.setFixedSize(QSize(64, 64))
        self.page_number.setAlignment(Qt.AlignCenter)
        self.page_number.setText(f"{self.current_page} / {self.n_pages}")

        #self.datacombo = QComboBox()
        #self.datacombo.setFixedSize(QSize(128, 30))
        #self.datacombo.addItems([str(i) for i in range(10)])
        #self.datacombo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        #self.load_box = QVBoxLayout()
        #self.load_box.addWidget(self.loadbutton)
        #self.load_box.addWidget(self.datacombo)
        key_box = QHBoxLayout()
        key_box.addWidget(self.selectallbutton)
        key_box.addWidget(self.selectnonebutton)
        key_box.addWidget(self.settingbutton)
        key_box.addWidget(self.prevbutton)
        key_box.addWidget(self.page_number)
        key_box.addWidget(self.nextbutton)
        key_box.addWidget(self.savebutton)
        key_box.addWidget(self.loadbutton)
        
        self.grid = QGridLayout()
        self.grid.setSpacing(1)
        
        main_box = QVBoxLayout()
        main_box.addLayout(self.grid)
        main_box.addLayout(key_box)

        main_widget = QWidget()
        main_widget.setLayout(main_box)
        self.setCentralWidget(main_widget)
        
        self.init_map()
        self.show()
    

    def calc_index(self, x, y):
        return((self.current_page - 1) * config['n_collage'] + x + \
               config['x_size'] * y)
    
    def get_image(self, id, mode):
        global images
        if mode == 'rgb':
            return(
                QImage(
                    images[id].data, self.imwidth, self.imheight,
                    self.imwidth * 3, QImage.Format_RGB888)
                )

    def get_label(self, id):
        global df
        return(df.label.iat[id])

    def init_map(self):
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
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
                
    def nextPage(self):
        if self.current_page < self.n_pages:
            self.current_page += 1
            print("Page:", self.current_page)
            self.page_number.setText(f"{self.current_page} / {self.n_pages}")
        else:
            print("This is the last page!")
        self.save_labels()
        self.reset_map()
        
    def prevPage(self):
        if self.current_page > 1:
            self.current_page -= 1
            print("Page:", self.current_page)
            self.page_number.setText(f"{self.current_page} / {self.n_pages}")
        else:
            print("This is the first page!")
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
                df.label.iat[w.id] = w.get_label()
                w.update()
                
        print("Selection: ", sum(df.label))

    def openSettings(self):
        self.settingWindow = SettingWindow()
        self.settingWindow.show()

    def load_data(self):
        global images
        global df
        self.file_path = self.dialog.getOpenFileName(
            self.loadbutton, "Open File", '',
            "HDF files (*.hdf5)")[0]
        self.file_name = os.path.basename(self.file_path)
        self.file_name = self.file_name.replace('.hdf5', '')

        # Load images
        with h5py.File(self.file_path, 'r') as file:
            if (config['image_key'] in file.keys() and 
                config['data_key'] in file.keys()):
                images = file[config['image_key']][:]
                df = pd.read_hdf(self.file_path, config['data_key'])
            else:
                sys.exit("HDF5 keys do not include given image and data keys!")


        images = channels2rgb8bit(images)
        self.n_events,self.imheight,self.imwidth,self.n_channels = images.shape
        
        self.n_pages = self.n_events // config['n_collage']

        print(
            'shape:', self.n_events, self.imheight, self.imwidth,
            self.n_channels, self.n_pages)

        if 'label' not in df.columns:
            df['label'] = np.zeros(self.n_events, dtype='uint8')

    def save_data(self, export_tsv=True):
        global df
        df.to_hdf(self.file_path, key=config['data_key'], index=False)
        if export_tsv:
            df.to_csv(
                f"{config['output_dir']}/{self.file_name}.txt", index=False,
                sep='\t')

    def closeEvent(self, event):
        with open('config.yml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)


# Functions
def load_config():
    global config
    if not os.path.exists('config.yml'):
        sys.exit("config file does not exist!")
    else:
        with open('config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        config['n_collage'] = config['x_size'] * config['y_size']
        config['tile_size'] = 2 * (config['tile_size'] // 2) + 1


def main():
    load_config()
    app = QApplication([])
    window = MainWindow()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == '__main__':
    main()
    
# Add input key-ID window
# Check if features data set exists
# Check if annotation_column exists
# Add multiple selection by dragging mouse click
# Add option 3-color or gray-scale
# Improve images
# Change it to dark theme
# Exit prompt


## Next Versions:
# Filter using size
