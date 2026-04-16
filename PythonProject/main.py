# main.py
import sys
from PyQt5.QtWidgets import QApplication
from GUI import AccentTranscriberApp  # import your GUI class

if __name__ == "__main__":
    app = QApplication(sys.argv)       # start the Qt application
    window = AccentTranscriberApp()    # create the main window
    window.show()                      # show the GUI
    sys.exit(app.exec_())              # run the app loop