#考勤信息查询
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QIcon, QPixmap
from PyQt5.QtCore import QCoreApplication, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QInputDialog
import threading
import cv2
import imutils
import os
import sys
# 导入数据库操作包
import pymysql
from datetime import datetime
# 添加数据库连接操作
from utils.GlobalVar import connect_to_sql


class InfoDialog(QWidget):
    def __init__(self):
        # super()构造器方法返回父级的对象。__init__()方法是构造器的一个方法。
        super().__init__()

        self.Dialog = InfoUI.Ui_Form()
        self.Dialog.setupUi(self)
        # 实现路径错误提示，方便定位错误
        self.current_filename = os.path.basename(__file__)
        # 设置查询信息按键连接函数
        self.Dialog.bt_check_info.clicked.connect(self.check_info)
        # 设置写入信息按键连接函数
        self.Dialog.bt_change_info.clicked.connect(self.change_info)
        # 设置查询班级人数按键的连接函数
        self.ui.bt_check.clicked.connect(self.check_nums)


    #“查询考勤信息”
    def check_info(self):
        # 获取用户输入的ID的内容，str格式
        self.input_id = self.Dialog.lineEdit_id.text()
        if self.input_id != '':
            # 用于存放统计信息
            lists = []
            # 打开数据库连接
            try:
                db, cursor = connect_to_sql()
            except ConnectionRefusedError as e:
                print("[ERROR] 数据库连接失败！", e)
            # 如果连接数据库成功，则继续执行查询
            else:
                # 查询语句，实现通过ID关键字检索个人信息的功能
                sql = "SELECT * FROM STUDENTS WHERE ID = {}".format(self.input_id)
                # 执行查询
                try:
                    cursor.execute(sql)
                    # 获取所有记录列表
                    results = cursor.fetchall()
                    for i in results:
                        lists.append(i[0])
                        lists.append(i[1])
                        lists.append(i[2])
                        lists.append(i[3])
                        lists.append(i[4])
                except ValueError as e:
                    print("[ERROR] 无法通过当前语句查询！", e)
                    
                else:
                    # 设置显示数据层次结构，5行2列(包含行表头)
                    table_view_module = QtGui.QStandardItemModel(5, 1)
                    # 设置数据行、列标题
                    table_view_module.setHorizontalHeaderLabels(['属性', '值'])
                    rows_name = ['学号', '姓名', '班级', '考勤时间','考勤状态']
           

                    # 设置填入数据内容
                    lists[0] = self.input_id
                    if len(lists) == 0:
                        QMessageBox.warning(self, "warning", "人脸数据库中无此人信息，请马上录入！", QMessageBox.Ok)
                    else:
                        for row, content in enumerate(lists):
                            row_name = QtGui.QStandardItem(rows_name[row])
                            item = QtGui.QStandardItem(content)
                            # 设置每个位置的行名称和文本值
                            table_view_module.setItem(row, 0, row_name)
                            table_view_module.setItem(row, 1, item)

                        # 指定显示的tableView控件，实例化表格视图
                        self.Dialog.tableView.setModel(table_view_module)
                    
                    assert isinstance(db, object)
                    # 关闭数据库连接
            finally:
                cursor.close()
                db.close()
    
    def check_dir_faces_num(self):
        num_dict = statical_facedata_nums()
        keys = list(num_dict.keys())
        values = list(num_dict.values())
        # print(values)
        # 如果没有人脸文件夹，则提示用户采集数据
        if len(keys) == 0:
            QMessageBox.warning(self, "Error", "face_dataset文件夹下没有人脸数据，请马上录入！", QMessageBox.Ok)
        else:
            # 设置显示数据层次结构，5行2列(包含行表头)
            table_view_module = QtGui.QStandardItemModel(len(keys), 1)
            table_view_module.setHorizontalHeaderLabels(['ID', 'Number'])

            for row, key in enumerate(keys):
                print(key, values[row])
                id = QtGui.QStandardItem(key)
                num = QtGui.QStandardItem(str(values[row]))

                # 设置每个位置的行名称和文本值
                table_view_module.setItem(row, 0, id)
                table_view_module.setItem(row, 1, num)

            # 指定显示的tableView控件，实例化表格视图
            self.Dialog.tableView.setModel(table_view_module)

    # 将采集信息写入数据库
    def write_info(self):
        # 存放信息的列表
        users = []
        # 信息是否完整标志位
        is_info_full = False
        student_id = self.Dialog.lineEdit_id.text()
        name = self.Dialog.lineEdit_name.text()
        which_class = self.Dialog.lineEdit_class.text()
        sex = self.Dialog.lineEdit_sex.text()
        birth = self.Dialog.lineEdit_birth.text()
        users.append((student_id, name, which_class, sex, birth))
        # 如果有空行，为False，则不执行写入数据库操作；反之为True，执行写入
        # Python内置函数all的作用是：如果用于判断的可迭代对象中全为True，则结果为True；反之为False
        if all([student_id, name, which_class, sex, birth]):
            is_info_full = True
        return is_info_full, users


    # “增加、修改考勤信息”
    def change_info(self):
        # 写入数据库
        try:
            db, cursor = connect_to_sql()
            # 如果存在数据，先删除再写入。前提是设置唯一索引字段或者主键。
            insert_sql = "replace into students(ID, Name, Class, Sex, Birthday) values(%s, %s, %s, %s, %s)"

            flag, users = self.write_info()
            if flag:
                cursor.executemany(insert_sql, users)
                QMessageBox.warning(self, "Warning", "修改成功，请勿重复操作！", QMessageBox.Ok)
            else:
                QMessageBox.information(self, "Error", "修改失败！请保证每个属性不为空！", QMessageBox.Ok)
        # 捕获所有除系统退出以外的所有异常
        except Exception as e:
            print("[ERROR] sql execute failed!", e)    
        finally:
            # 提交到数据库执行
            db.commit()
            # 关闭数据库
            cursor.close()
            # 关闭数据库连接
            db.close()


# 查询班级人数
    def check_nums(self):
        # 选择的班级
        input_class = self.ui.comboBox_class.currentText()
        # print("[INFO] 你当前选择的班级为:", input_class)
        if input_class != '':
            try:
                # 打开数据库连接, 使用cursor()方法获取操作游标
                db, cursor = connect_to_sql()
            except ValueError:
                self.ui.textBrowser_log.append("[ERROR] 连接数据库失败！")
            else:
                self.ui.textBrowser_log.append("[INFO] 连接数据库成功，正在执行查询...")
                # 查询语句，实现通过ID关键字检索个人信息的功能
                sql = "select * from studentnums where class = {}".format(input_class)
                cursor.execute(sql)
                # 获取所有记录列表
                results = cursor.fetchall()
                self.nums = []
                for i in results:
                    self.nums.append(i[1])
                    
                # 用于查询每班的实到人数
                sql2 = "select * from checkin where class = {}".format(input_class)
                cursor.execute(sql2)
                
                # 获取所有记录列表
                results2 = cursor.fetchall()
                self.ui.textBrowser_log.append("[INFO] 查询成功！")
                
                # 关闭数据库连接
                db.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 创建并显示窗口
    mainWindow = MainWindow()
    infoWindow = InfoDialog()
    mainWindow.ui.bt_gathering.clicked.connect(infoWindow.handle_click)
    mainWindow.show()
    sys.exit(app.exec_())
