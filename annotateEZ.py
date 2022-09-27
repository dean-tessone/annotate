from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pandas as pd
import h5py
import numpy as np
import tables

# Input
hdf5_path = sys.argv[1]
TILE_SIZE = 43
COLLAGE_SIZE = (int(sys.argv[2]), int(sys.argv[3]))
n_collage = COLLAGE_SIZE[0] * COLLAGE_SIZE[1]

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
    
w_index = lambda x,y : (current_page - 1) * n_collage + x + COLLAGE_SIZE[0] * y

# Load images
with h5py.File(hdf5_path, 'r') as file:
    images = file['images'][:]
df = pd.read_hdf(hdf5_path, 'features')


images = channels2rgb8bit(images)
N, height, width, channel = images.shape
bytesPerLine = 3 * width
n_pages = N // n_collage
current_page = 1


if 'label' not in df.columns:
    df['label'] = np.zeros(N, dtype='bool')


print('shape: ', N, height, width, channel, n_pages)


# Classes
class Pos(QWidget):
    
    def __init__(self, x, y, *args, **kwargs):
        super(Pos, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(TILE_SIZE, TILE_SIZE))
        self.x = x
        self.y = y
        self.image = QImage(images[w_index(self.x, self.y)].data, width,
                            height, bytesPerLine, QImage.Format_RGB888)
        self.is_interesting = df.label.iat[w_index(self.x, self.y)]
        
    def reset(self):
        self.is_interesting = df.label.iat[w_index(self.x, self.y)]
        self.image = QImage(images[w_index(self.x, self.y)].data, width,
                            height, bytesPerLine, QImage.Format_RGB888)
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        r = event.rect()
        p.drawImage(r, self.image)
        # p.drawPixmap(r, QPixmap(self.image))
        if self.is_interesting:
            color = Qt.red
        else:
            color = Qt.black
        pen = QPen(color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r)
        
    def flag(self):
        self.is_interesting = True
        print(f"Event ({self.x}, {self.y}) flagged!")
        self.update()
        
    def junk(self):
        self.is_interesting = False
        print(f"Event ({self.x}, {self.y}) junked!")
        self.update()
    
    def savelabel(self):
        df.label.iat[w_index(self.x, self.y)] = self.is_interesting
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.junk()
        elif event.button() == Qt.LeftButton:
            self.flag()


class MainWindow(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.x_size, self.y_size = COLLAGE_SIZE
        
        w = QWidget()
        vb = QVBoxLayout()
        
        self.selectallbutton = QPushButton("All")
        self.selectallbutton.setFixedSize(QSize(64, 64))
        self.selectallbutton.setIconSize(QSize(32, 32))
        self.selectallbutton.pressed.connect(self.selectAll)
        
        self.selectnonebutton = QPushButton("None")
        self.selectnonebutton.setFixedSize(QSize(64, 64))
        self.selectnonebutton.setIconSize(QSize(32, 32))
        self.selectnonebutton.pressed.connect(self.selectNone)
        
        self.nextbutton = QPushButton()
        self.nextbutton.setFixedSize(QSize(64, 64))
        self.nextbutton.setIconSize(QSize(32, 32))
        self.nextbutton.setIcon(QIcon("./icons/Right.png"))
        self.nextbutton.pressed.connect(self.nextPage)
        
        self.prevbutton = QPushButton()
        self.prevbutton.setFixedSize(QSize(64, 64))
        self.prevbutton.setIconSize(QSize(32, 32))
        self.prevbutton.setIcon(QIcon("./icons/Left.png"))
        self.prevbutton.pressed.connect(self.prevPage)
        
        self.savebutton = QPushButton()
        self.savebutton.setFixedSize(QSize(64, 64))
        self.savebutton.setIconSize(QSize(32, 32))
        self.savebutton.setIcon(QIcon("./icons/Save.png"))
        self.savebutton.pressed.connect(self.saveLabels)
        
        vb.addWidget(self.selectallbutton)
        vb.addWidget(self.selectnonebutton)
        vb.addWidget(self.nextbutton)
        vb.addWidget(self.prevbutton)
        vb.addWidget(self.savebutton)
        
        hb = QHBoxLayout()
        
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        hb.addLayout(self.grid)
        hb.addLayout(vb)
        w.setLayout(hb)
        self.setCentralWidget(w)
        
        self.init_map()
        self.show()
        
    def init_map(self):
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
                w = Pos(x, y)
                self.grid.addWidget(w, y, x)

    def reset_map(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.reset()
                
    def nextPage(self):
        global current_page
        if current_page < n_pages:
            current_page = current_page + 1
            print("Page:", current_page)
        else:
            print("This is the last page!")
        self.reset_map()
        
    def prevPage(self):
        global current_page
        if current_page > 1:
            current_page = current_page - 1
            print("Page:", current_page)
        else:
            print("This is the first page!")
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
                
    def saveLabels(self):
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.savelabel()
                w.update()
                
        print("Selection: ", sum(df.label))
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)    
    window = MainWindow()
    ret = app.exec_()
    df.to_csv(hdf5_path.replace("hdf5", "txt"), sep='\t', index=True)
    sys.exit(ret)

# Save data
# Filter using size
# Add multichannel capability
# Add input key-ID window
# Check if features data set exists
# Check if annotation_column exists
# Add export of features
# Add multiple selection by dragging mouse click
