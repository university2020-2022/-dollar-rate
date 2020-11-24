import re
import requests 
from bs4 import BeautifulSoup
import time
import sqlite3 as sql
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import asarray
import os
import tensorflow as tf


from PyQt5 import QtWidgets, uic
from test import Ui_MainWindow
link = "https://myfin.by/currency/minsk"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
p = re.compile(r"<td>(.*?)</td>")
con = sql.connect('main.db')

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.textBrowser.setText("PyScripts")
        self.ui.listWidget.addItem("PyScripts")
        self.ui.pushButton.clicked.connect(self.test)
    def test(self):
        self.ui.listWidget.clear()
        a = start()
        b = "Белинвестбанк:", a[0] , "Нейронная сеть:", a[1], "Альфа-Банк:", a[2] , "Нейронная сеть:", a[3], "Нацбанк:", a[4] , "Нейронная сеть:", a[5]
        self.ui.textBrowser.setText(str(b))
        
        cur = con.cursor()
        cur.execute("SELECT * FROM `test`")
        rows = cur.fetchall()
        for row in rows:
            if self.ui.checkBox.isChecked():
                self.ui.listWidget.addItem("Белинвестбанк: "+ str(row[0])+";    Прогноз нейронной сети: "+ str("%.3f" % row[1]))
            if self.ui.checkBox_2.isChecked():
                self.ui.listWidget.addItem("Альфа-Банк: "+ str(row[2])+";         Прогноз нейронной сети: "+ str("%.2f" % row[3]))
            if self.ui.checkBox_3.isChecked():
                self.ui.listWidget.addItem("Нацбанк: "+str(row[4])+";               Прогноз нейронной сети: "+ str("%.2f" % row[3]))
                
          
        


















def save(bel, bel1, alf, alf1, vtb, vtb1, time):
 #print(bel, bel1, alf, alf1, vtb, vtb1, time)
 with con:
  cur = con.cursor()
  cur.execute("CREATE TABLE IF NOT EXISTS `test` (`bel` INTEGER, `bel1` INTEGER, `alf` INTEGER, `alf1` INTEGER, `vtb` INTEGER, `vtb1` INTEGER, `time` INTEGER)")
  #cur.execute(f"INSERT INTO `test` VALUES ('{bel}', '{bel1}', '{alf}', '{alf1}', '{vtb}', '{vtb1}', '{time}')")
  cur.execute("SELECT * FROM `test`")
  rows = cur.fetchall()
  print(rows[-1][0], rows[-1][2], rows[-1][4] , bel, alf, vtb )
  if float(rows[-1][0]) == float(bel) and float(rows[-1][2]) == float(alf) and float(rows[-1][4]) == float(vtb):
      print("ПОВТОР")
  else:
      cur.execute(f"INSERT INTO `test` VALUES ('{bel}', '{bel1}', '{alf}', '{alf1}', '{vtb}', '{vtb1}', '{time}')")
      print("СОХРАНИЛИ")
      
  #for row in rows:
  # print(row[0], row[1], row[2], row[3], row[4], row[5], row[6])


def load(curs, bank):

  x_train0 = np.load('x_train0.npy')
  mean = x_train0.mean(axis=0)
  std = x_train0.std(axis=0)
  x_train0 -= mean
  x_train0 /= std
  
  model = Sequential()
  checkpoint_path = "cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Создаем коллбек сохраняющий веса модели
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
  model.add(Dense(128, activation='relu', input_shape=(x_train0.shape[1],)))
  model.add(Dense(1))
  model.compile(optimizer="adam", loss="mse", metrics=['mae'])
  model.load_weights(checkpoint_path)
###############################################################################################################################
  cur = con.cursor()
  cur.execute("CREATE TABLE IF NOT EXISTS `test` (`bel` INTEGER, `bel1` INTEGER, `alf` INTEGER, `alf1` INTEGER, `vtb` INTEGER, `vtb1` INTEGER, `time` INTEGER)")
  cur.execute("SELECT * FROM `test`")
  rows = cur.fetchall()
  ar =[]
  for row in rows:
   ar.append(row[bank])
   #print(row[0])
  ar.append(curs)
 
  tes = np.fromiter(ar[-13:], dtype=float)
  
  pred = model.predict(np.array([tes]))
  return pred
    
def start():    
  
 
# Парсим всю страницу
  full_page = requests.get(link, headers=headers).text
  soup = BeautifulSoup(full_page, "html.parser")
  price1 = soup.find_all('tr', {'class': 'tr-tb acc-link_14 not_h'})[0]
#Белинвестбанк
  price11 = re.findall(p, str(price1))[0]
  belb = load(re.findall(p, str(price1))[0], 0)

# soup = BeautifulSoup(full_page, "html.parser")
  price2 = soup.find_all('tr', {'class': 'tr-tb acc-link_6 not_h'})[0]
#Альфа-Банк
  price22 = re.findall(p, str(price2))[0]
  alfb = load(re.findall(p, str(price2))[0], 3)

 
# soup = BeautifulSoup(full_page, "html.parser")
  price3 = soup.find_all('tr', {'class': 'tr-tb acc-link_8 not_h'})[0]
#Банк ВТБ
  price33 = re.findall(p, str(price3))[0]
  vtbb = load(re.findall(p, str(price3))[0], 5)
  print(price11, belb[0][0], price22, alfb[0][0], price33, vtbb[0][0])
  save(price11, belb[0][0], price22, alfb[0][0], price33, vtbb[0][0], datetime.datetime.now())

  return price11, belb[0][0], price22, alfb[0][0], price33, vtbb[0][0]
 
app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())

