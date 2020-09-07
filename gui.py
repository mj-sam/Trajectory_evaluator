import sys, csv
import numpy
import random

from gui_base import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import numpy as np
from utils import align,read_file_list,associate,plot_traj
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.pyplot as plt 

class Evaluator(QObject):
    finished = pyqtSignal()  # give worker class a finished signal
    def __init__(self, main_self,parent=None):
        QObject.__init__(self, parent=parent)
        self.continue_run = True  # provide a bool run condition for the class
        self.aligned_error = None
        self.main_self = main_self
    
    def evaluation(self):
        if(self.main_self.prediction_path != None and self.main_self.ground_truth_path != None):
            self.main_self.scale_value = float(self.main_self.scale_input.text())
            self.main_self.max_difrence= float(self.main_self.max_input.text())
            
            first_list = read_file_list(self.main_self.ground_truth_path)
            second_list = read_file_list(self.main_self.prediction_path)
            matches = associate(first_list, second_list,float(0.0),self.main_self.max_difrence)
            first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()

            second_xyz = numpy.matrix([[float(value)*float(self.main_self.scale_value) for value in second_list[b][0:3]] for a,b in matches]).transpose()


            rot,trans,trans_error, self.alignment_error = align(second_xyz,first_xyz)
            second_xyz_aligned = rot * second_xyz + trans

            first_stamps = first_list.keys()
            first_stamps = list(first_stamps)
            first_stamps.sort()
            first_xyz_full = numpy.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

            second_stamps = second_list.keys()
            second_stamps = list(second_stamps)
            second_stamps.sort()

            second_xyz_full = numpy.matrix([[float(value)*float(self.main_self.scale_value) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
            second_xyz_full_aligned = rot * second_xyz_full + trans

            #if args.verbose:
            print( "compared_pose_pairs %d pairs"%(len(trans_error)))

            self.main_self.rmse_res.setText(str(numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))))
            self.main_self.mean_res.setText(str(numpy.mean(trans_error)))
            self.main_self.median_res.setText(str(numpy.median(trans_error)))
            self.main_self.min_res.setText(str(numpy.min(trans_error)))
            self.main_self.max_res.setText(str(numpy.max(trans_error)))
            self.main_self.std_res.setText(str(numpy.std(trans_error)))


            self.main_self.figure.clear() 
            ax = self.main_self.figure.add_subplot(111) 
            #print("shape : ",self.alignment_error[0,:].shape)
            #ax.boxplot(self.alignment_error[0,:])

            
            plot_traj(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","ground truth")
            plot_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","estimated")
            label="difference"
            for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A):
                ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
                label=""

            ax.legend()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            
            self.main_self.canvas.draw() 

        self.main_self.eval_butt.setEnabled(True)
        self.main_self.eval_butt.setText('Evaluate')
        self.finished.emit()  # emit the finished signal when the loop is done

    def stop(self):
        print('finished')
        self.continue_run = False  # set the run condition to false on stop

class gui(Ui_MainWindow):
    def __init__(self,MainWindow):
        super(Ui_MainWindow,gui).__init__(self)
        self.setupUi(MainWindow)   
        #self.retranslateUi(MainWindow)
        self.clear_result()
        self.pred_butt.clicked.connect(self.Pred_file_set)
        self.gt_butt.clicked.connect(self.gt_file_set)
        self.Add_butt.clicked.connect(self.Add_row)
        self.save_but.clicked.connect(self.handleSave)
        self.load_but.clicked.connect(self.handleOpen)
        self.Plt_bar.clicked.connect(self.handleBarClicked)
        self.plt_box.clicked.connect(self.handleBoxClicked)
        self.main_window = MainWindow
        self.eval_butt.clicked.connect(self.eval_click)
        self.figure = plt.figure() 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, None)
        self.vertical_canvas.addWidget(self.toolbar)
        self.vertical_canvas.addWidget(self.canvas) 
        #self.setLayout(self.vertical_canvas) 
        #self.eval_butt.setText('test')
        # Thread:
        self.thread = QThread()
        self.Evaluator = Evaluator(self)
        self.Evaluator.moveToThread(self.thread)

        self.Evaluator.finished.connect(self.thread.quit)  # connect the workers finished signal to stop thread
        self.Evaluator.finished.connect(self.Evaluator.deleteLater)  # connect the workers finished signal to clean up worker
        self.thread.finished.connect(self.thread.deleteLater)  # connect threads finished signal to clean up thread
        
        #self.plot()
        self.mutex_eval = True
        self.ground_truth_path = None
        self.prediction_path   = None
        self.res_table.horizontalHeader().setStretchLastSection(True)
        self.res_table.setColumnCount(9)
        self.res_table.setRowCount(0)
        self.res_table.setHorizontalHeaderLabels(['Plot','Dataset', 'Method', 'RMSE', 'Median', 'Mean','Min', 'Max', 'Std'])        

    def Add_row(self):
        rowPosition = self.res_table.rowCount()
        self.res_table.insertRow(rowPosition)
        item = QtWidgets.QTableWidgetItem('')
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Unchecked)
        self.res_table.setItem(rowPosition , 0, item)
        self.res_table.setItem(rowPosition , 1, QtWidgets.QTableWidgetItem(self.dataset_input.text()))
        self.res_table.setItem(rowPosition , 2, QtWidgets.QTableWidgetItem(self.method_input.text()))
        self.res_table.setItem(rowPosition , 3, QtWidgets.QTableWidgetItem(self.rmse_res.text()))
        self.res_table.setItem(rowPosition , 4, QtWidgets.QTableWidgetItem(self.median_res.text()))
        self.res_table.setItem(rowPosition , 5, QtWidgets.QTableWidgetItem(self.mean_res.text()))
        self.res_table.setItem(rowPosition , 6, QtWidgets.QTableWidgetItem(self.min_res.text()))
        self.res_table.setItem(rowPosition , 7, QtWidgets.QTableWidgetItem(self.max_res.text()))
        self.res_table.setItem(rowPosition , 8, QtWidgets.QTableWidgetItem(self.std_res.text()))
        #action called by the push button
    
    def plot(self): 
        data = [random.random() for i in range(10)] 
        self.figure.clear() 
        ax = self.figure.add_subplot(111) 
        ax.plot(data, '*-') 
        self.canvas.draw() 

    def clear_result(self):
        _translate = QtCore.QCoreApplication.translate
        self.pred_path.setText(_translate("MainWindow", " Trajectory path has not been set"))
        self.gt_path.setText(_translate("MainWindow", " Ground truth path has not been set"))
        self.scale.setText(_translate("MainWindow", "Scale "))
        self.label_2.setText(_translate("MainWindow", "Max diffrence"))
        
        self.rmse_res.setText(_translate("MainWindow", " None "))
        self.mean_res.setText(_translate("MainWindow", " None "))
        self.median_res.setText(_translate("MainWindow", " None "))
        self.std_res.setText(_translate("MainWindow", " None "))
        self.min_res.setText(_translate("MainWindow", " None "))
        self.max_res.setText(_translate("MainWindow", " None "))

    def Pred_file_set(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                            None,
                            caption='Open file',
                            directory='~',
                            filter='*.csv *.txt')
        if filename:
            self.pred_path.setText(filename)
            self.prediction_path = filename
    
    def gt_file_set(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None,directory='~',caption='Open file',filter='*.csv *.txt')
        if filename:
            self.gt_path.setText(filename)
            self.ground_truth_path = filename

    def eval_click(self):
        self.eval_butt.setText('Please wait')
        self.eval_butt.setEnabled(False)
        self.Evaluator.evaluation()
        
    def handleSave(self):
        path , ext = QtWidgets.QFileDialog.getSaveFileName(None, 'Save File', '', 'CSV(*.csv)')
        print(len(path))
        if len(path)>1:
            with open(str(path+'.csv'), 'w') as stream:
                writer = csv.writer(stream)
                for row in range(self.res_table.rowCount()):
                    rowdata = []
                    for column in range(1,self.res_table.columnCount()):
                        item = self.res_table.item(row, column)
                        if item is not None:
                            rowdata.append(str(item.text()))
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)

    def handleBarClicked(self):
        data_set = set()
        method  = set()

        for i in range(self.res_table.rowCount()):
            item = self.res_table.item(i, 0)
            if item.checkState() == QtCore.Qt.Checked:
                data_set.add(self.res_table.item(i, 1).text())
                method.add(self.res_table.item(i, 2).text())

        
        data_set = list(data_set)
        #data_set = ['z', 'x', 'w', 'y', 'q', 'p']
        dic_data_set = {}
        for i in range(len(data_set)):
            dic_data_set[data_set[i]] = i
            
        method = list(method)
        #method = ['c', 'b', 'a']
        dic_method = {}
        for i in range(len(method)):
            dic_method[method[i]] = i

        if(len(data_set) > 0 and len(method) > 0):
            print('dic :',dic_method)
            print('dic :',dic_data_set)
            result = [[0 for i in range(len(data_set))] for i in range(len(method))]
            #result = [[30, 25, 50,40, 23, 51],[35, 22, 45,42, 13, 41],[40, 23, 51,40, 23, 51],]
            print(result)
            for i in range(self.res_table.rowCount()):
                item = self.res_table.item(i, 0)
                if item.checkState() == QtCore.Qt.Checked: 
                    data_name = self.res_table.item(i, 1).text()
                    method_name = self.res_table.item(i, 2).text()
                    result[dic_method[method_name]][dic_data_set[data_name]] = float(self.res_table.item(i, 3).text())
                
            self.figure.clear() 
            print(result)
            X = np.arange(len(result[0]))
            ax = self.figure.add_subplot(111) 

            colors = ['b','g','r','c','m','y','k']
            bar_width = 1.0/(len(result)+1)
            bar_shift = [0.0]
            for i in range(len(result)):
                bar_shift.append(bar_shift[-1]+bar_width)
            for ax_bar in range(len(result)):
                ax.bar(X + bar_shift[ax_bar], result[ax_bar], color = colors[ax_bar], width = bar_width,label=method[ax_bar])
            ax.set_ylabel('RMSE value')
            ax.set_title('RMSE by data set and method')
            ax.set_xticks(X)
            ax.set_xticklabels(data_set)
            ax.legend()
            # ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
            # ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
            # ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
            #ax.plot(data, '*-')
            self.canvas.draw() 


    def handleBoxClicked(self):
        data_set = set()
        method  = set()

        for i in range(self.res_table.rowCount()):
            item = self.res_table.item(i, 0)
            if item.checkState() == QtCore.Qt.Checked:
                data_set.add(self.res_table.item(i, 1).text())
                method.add(self.res_table.item(i, 2).text())

        
        data_set = list(data_set)
        #data_set = ['z', 'x', 'w', 'y', 'q', 'p']
        dic_data_set = {}
        for i in range(len(data_set)):
            dic_data_set[data_set[i]] = i
            
        method = list(method)
        #method = ['c', 'b', 'a']
        dic_method = {}
        for i in range(len(method)):
            dic_method[method[i]] = i

        if(len(data_set) > 0 and len(method) > 0):
            print('dic :',dic_method)
            print('dic :',dic_data_set)
            result = [[
                {'med': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0,'rmse':0} 
                for i in range(len(data_set))] for i in range(len(method))]
            #result = [[30, 25, 50,40, 23, 51],[35, 22, 45,42, 13, 41],[40, 23, 51,40, 23, 51],]
            print(result)
            for i in range(self.res_table.rowCount()):
                item = self.res_table.item(i, 0)
                if item.checkState() == QtCore.Qt.Checked: 
                    data_name = self.res_table.item(i, 1).text()
                    method_name = self.res_table.item(i, 2).text()
                    result[dic_method[method_name]][dic_data_set[data_name]]['rmse'] = float(self.res_table.item(i, 3).text())
                    result[dic_method[method_name]][dic_data_set[data_name]]['med'] = float(self.res_table.item(i, 4).text())
                    Q = float(self.res_table.item(i, 8).text()) #* 0.6745
                    Mean = float(self.res_table.item(i, 5).text())
                    result[dic_method[method_name]][dic_data_set[data_name]]['q1'] = Mean - Q
                    result[dic_method[method_name]][dic_data_set[data_name]]['q3'] = Mean + Q
                    result[dic_method[method_name]][dic_data_set[data_name]]['mean'] = Mean
                    result[dic_method[method_name]][dic_data_set[data_name]]['whislo'] = float(self.res_table.item(i, 6).text())
                    result[dic_method[method_name]][dic_data_set[data_name]]['whishi'] = float(self.res_table.item(i, 7).text())  

            self.figure.clear() 
            print(result)
            X = np.arange(len(result[0]))
            ax = self.figure.add_subplot(111) 

            colors = ['b','g','r','c','m','y','k']
            bar_width = 1.0/(len(result)+1)
            bar_shift = [0.0]
            for i in range(len(result)):
                bar_shift.append(bar_shift[-1]+bar_width)
            for ax_bar in range(len(result)):
                ax.bxp(result[ax_bar], showfliers=False)
                #ax.bar(X + bar_shift[ax_bar], result[ax_bar], color = colors[ax_bar], width = bar_width,label=method[ax_bar])
            ax.set_ylabel('RMSE value')
            ax.set_title('RMSE by data set and method')
            ax.set_xticks(X)
            ax.set_xticklabels(data_set)
            ax.legend()
            # ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
            # ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
            # ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
            #ax.plot(data, '*-')
            self.canvas.draw() 


    def handleOpen(self):
        path =  QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '', 'CSV(*.csv)')
        #print(path)
        if path[0]!='':
            with open(str(path[0]), 'r') as stream:
                self.res_table.setRowCount(0)
                #self.res_table.setColumnCount(0)
                for rowdata in csv.reader(stream):
                    #row = self.res_table.rowCount()
                    rowPosition = self.res_table.rowCount()
                    item = QtWidgets.QTableWidgetItem('')
                    item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    
                    self.res_table.insertRow(rowPosition)
                    self.res_table.setItem(rowPosition , 0, item)
                    self.res_table.setItem(rowPosition , 1, QtWidgets.QTableWidgetItem(str(rowdata[0])))
                    self.res_table.setItem(rowPosition , 2, QtWidgets.QTableWidgetItem(str(rowdata[1])))
                    self.res_table.setItem(rowPosition , 3, QtWidgets.QTableWidgetItem(str(rowdata[2])))
                    self.res_table.setItem(rowPosition , 4, QtWidgets.QTableWidgetItem(str(rowdata[3])))
                    self.res_table.setItem(rowPosition , 5, QtWidgets.QTableWidgetItem(str(rowdata[4])))
                    self.res_table.setItem(rowPosition , 6, QtWidgets.QTableWidgetItem(str(rowdata[5])))
                    self.res_table.setItem(rowPosition , 7, QtWidgets.QTableWidgetItem(str(rowdata[6])))
                    self.res_table.setItem(rowPosition , 8, QtWidgets.QTableWidgetItem(str(rowdata[7])))
                    '''
                    self.res_table.insertRow(row)
                    self.res_table.setColumnCount(len(rowdata))
                    for column, data in enumerate(rowdata):
                        item = QtGui.QTableWidgetItem(data)
                        self.res_table.setItem(row, column, item)
                    '''


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    #MainWindow.
    ui = gui(MainWindow)
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



