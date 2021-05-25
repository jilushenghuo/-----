#!/usr/bin/env python3

import dlib
import pymysql
import telegram
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox, QAbstractItemView, QTableWidgetItem
from PyQt5.uic import loadUi

import os
import webbrowser
import logging
import logging.config
import sys
import threading
import queue
import multiprocessing
import winsound
import numpy
import csv

from configparser import ConfigParser
from datetime import datetime

from dataRecord import DataRecordUI

fontStyle = ImageFont.truetype(
    "微软雅黑Bold.ttf", 20, encoding="utf-8")  # 字体格式

haar_eyes_cascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')  # 眼部识别
haar_smile_cascade = cv2.CascadeClassifier('./haar/haarcascade_smile.xml')  # 微笑识别
predictor_5 = dlib.shape_predictor('./shape_predictor/shape_predictor_5_face_landmarks.dat')  # 5特征点模型
predictor_68 = dlib.shape_predictor('./shape_predictor/shape_predictor_68_face_landmarks.dat')  # 68特征点模型
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # 人脸识别器模型
dlib_detector = dlib.get_frontal_face_detector()  # dlib 人脸检测器

def recognition(self):

    self.isRunning = True
    # 帧数、人脸ID初始化
    frameCounter = 0
    currentFaceID = 0

    # 人脸跟踪器字典初始化
    faceTrackers = dict()
    all_stu_features = []

    isTrainingDataLoaded = False  # 预加载训练数据标记，检查一次过后即可不检查
    isDbConnected = False  # 预连接数据库标记，连接一次后即可只检查标记

    while self.isRunning and CoreUI.cap.isOpened():
        ret, frame = CoreUI.cap.read()  # 从摄像头捕获帧

        # 预加载识别数据
        if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):  # 训练数据
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # LBPH人脸识别对象
            recognizer.read(CoreUI.trainingData)  # 读取训练数据
            if os.path.exists(CoreUI.dlib_features_data):
                csv_reader = pd.read_csv(CoreUI.dlib_features_data, header=None)
                all_stu_features = self.read_dlib_features_csv(csv_reader)

            isTrainingDataLoaded = True

        if not isDbConnected:  # 学生信息数据库
            conn, cursor = connect_to_sql()
            isDbConnected = True

        captureData = {}  # 单帧识别结果
        realTimeFrame = frame.copy()  # copy原始帧的识别帧
        alarmSignal = {}  # 报警信号

        # haar级联分类器+LBPH局部二值模式识别方法
        if self.is_haar_faceCascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度图，改变颜色空间，实际上就是BGR三色2(to) 灰度GRAY
            faces = self.find_faces_by_haar(frame)  # 分类器获取人脸



            # 人脸跟踪
            # Reference：https://github.com/gdiepen/face-recognition
            if self.isFaceTrackerEnabled:
                know_faces = set()

                # 删除质量较低的跟踪器
                self.del_low_quality_face_tracker(faceTrackers, realTimeFrame)

                # 遍历所有侦测到的人脸坐标
                for (_x, _y, _w, _h) in faces:

                    # 微笑检测
                    smiles_y = (_y + _h + _y) // 2
                    smiles = haar_smile_cascade.detectMultiScale(gray[smiles_y:_y + _h, _x:_x + _w], 1.3, 10,
                                                                 minSize=(10, 10))
                    for (x, y, w, h) in smiles:
                        cv2.rectangle(realTimeFrame, (_x + x, smiles_y + y), (_x + x + w, smiles_y + y + h),
                                      (200, 50, 0), 1)
                        break

                    # 眼部检测
                    eyes = haar_eyes_cascade.detectMultiScale(gray[_y:smiles_y, _x:_x + _w], 1.3, 10,
                                                              minSize=(40, 40))
                    for (x, y, w, h) in eyes:
                        cv2.rectangle(realTimeFrame, (_x + x, _y + y), (_x + x + w, _y + y + h), (150, 255, 30), 1)

                    isKnown = False

                    # 人脸识别
                    if self.isFaceRecognizerEnabled:
                        # 蓝色识别框（RGB三通道色参数其实顺序是BGR）
                        cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (232, 138, 30), 2)
                        # 预测函数，识别后返回face ID和差异程度，差异程度越小越相似
                        face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])
                        logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                        if self.isDebugMode:  # 调试模式输出每帧识别信息
                            CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                        # 从数据库中获取识别人脸的身份信息
                        try:
                            cursor.execute("SELECT * FROM users WHERE face_id=%s", (face_id,))
                            result = cursor.fetchall()
                            if result:
                                stu_id = str(result[0][0])  # 学号
                                zh_name = result[0][2]  # 中文名
                                en_name = result[0][3]  # 英文名
                            else:
                                raise Exception
                        except Exception as e:
                            logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                            CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                            stu_id = ''
                            zh_name = ''
                            en_name = ''

                        # 若置信度评分小于置信度阈值，认为是可靠识别
                        if confidence < self.confidenceThreshold:
                            isKnown = True
                            if self.isPanalarmEnabled:  # 签到系统启动状态下执行
                                stu_statu = self.attendance_list.get(stu_id, 0)
                                if stu_statu > 9:
                                    realTimeFrame = cv2ImgAddText(realTimeFrame, '已识别', _x + _w - 45, _y - 10,
                                                                  (0, 97, 255))  # 帧签到状态标记
                                elif stu_statu <= 8:
                                    # 连续帧识别判断，避免误识
                                    self.attendance_list[stu_id] = stu_statu + 1
                                else:
                                    attendance_time = datetime.now()
                                    self.attendance_list[stu_id] = stu_statu + 1
                                    alarmSignal = {
                                        'id': stu_id,
                                        'name': zh_name,
                                        'time': attendance_time,
                                        'img': realTimeFrame
                                    }
                                    CoreUI.attendance_queue.put(alarmSignal)  # 签到队列插入该信号
                                    logging.info('系统发出了新的签到信号')
                            # 置信度标签
                            cv2.putText(realTimeFrame, str(round(100 - confidence, 3)), (_x - 5, _y + _h + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 1)
                            # 蓝色英文名标签
                            cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 97, 255), 2)
                            # 蓝色中文名标签
                            realTimeFrame = cv2ImgAddText(realTimeFrame, zh_name, _x - 5, _y - 10, (0, 97, 255))

                            know_faces.add(stu_id)
                            if self.isDebugMode:  # 调试模式输出每帧识别信息
                                print(know_faces)
                                print(self.attendance_list)
                        else:
                            cv2.putText(realTimeFrame, str(round(100 - confidence, 3)), (_x - 5, _y + _h + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 50, 255), 1)
                            # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                            cv2.putText(realTimeFrame, 'unknown', (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            # 若置信度评分超出自动报警阈值，触发报警信号
                            if confidence > self.autoAlarmThreshold:
                                # 报警系统是否开启
                                if self.isPanalarmEnabled:  # 记录报警时间戳和当前帧
                                    alarmSignal['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')
                                    alarmSignal['img'] = realTimeFrame
                                    CoreUI.alarmQueue.put(alarmSignal)  # 报警队列插入该信号
                                    logging.info('系统发出了未知人脸信号')

                    # 帧数计数器
                    frameCounter += 1

                    # 每读取10帧，更新检测跟踪器的新增人脸
                    if frameCounter % 10 == 0:
                        frameCounter = 0  # 防止爆int
                        # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                        # 而dlib人脸跟踪器要求的是int类型
                        x, y, w, h = int(_x), int(_y), int(_w), int(_h)

                        # 计算中心点
                        x_bar = x + 0.5 * w
                        y_bar = y + 0.5 * h

                        # matchedFid表征当前检测到的人脸是否已被跟踪，未赋值则
                        matchedFid = None

                        # 遍历人脸追踪器的face_id
                        for fid in faceTrackers.keys():
                            # 获取人脸跟踪器的位置
                            # tracked_position 是 dlib.drectangle 类型，用来表征图像的矩形区域，坐标是浮点数
                            tracked_position = faceTrackers[fid].get_position()
                            # 浮点数取整
                            t_x = int(tracked_position.left())
                            t_y = int(tracked_position.top())
                            t_w = int(tracked_position.width())
                            t_h = int(tracked_position.height())

                            # 计算人脸跟踪器的中心点
                            t_x_bar = t_x + 0.5 * t_w
                            t_y_bar = t_y + 0.5 * t_h

                            # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                            # 说明当前人脸已被跟踪
                            if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                    (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                matchedFid = fid

                        # 如果当前检测到的人脸是陌生人脸且未被跟踪
                        if not isKnown and matchedFid is None:
                            # 创建一个追踪器
                            tracker = dlib.correlation_tracker()  # 多目标追踪器
                            # 设置图片中被追踪物体的范围，也就是一个矩形框
                            tracker.start_track(realTimeFrame, dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                            # 将该人脸跟踪器分配给当前检测到的人脸
                            faceTrackers[currentFaceID] = tracker
                            # 人脸ID自增
                            currentFaceID += 1


                for fid in faceTrackers.keys():
                    tracked_position = faceTrackers[fid].get_position()

                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    # 在跟踪帧中绘制方框圈出人脸，红框
                    cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                    # 图像/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                    cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                1)
                del_list = []
                for stu_id, value in self.attendance_list.items():
                    if stu_id not in know_faces and value <= 8:
                        del_list.append(stu_id)
                for stu_id in del_list:
                    self.attendance_list.pop(stu_id, 0)

        else:
            # dlib人脸关键点识别,绿框
            face_rects = self.find_faces_by_dlib(frame)
            # print(face_rects, scores, idx)  # rectangles[[(281, 209) (496, 424)]] [0.07766452427949444] [4]

            if self.isFaceTrackerEnabled:
                know_faces = set()
                # 删除质量较低的跟踪器
                # self.del_low_quality_face_tracker(faceTrackers, realTimeFrame)

                # 遍历所有侦测到的人脸坐标
                for rect in face_rects:

                    isKnown = False

                    left = rect.left()
                    top = rect.top()
                    right = rect.right()
                    bottom = rect.bottom()

                    # 绘制出侦测人臉的矩形范围,绿框
                    cv2.rectangle(realTimeFrame, (left - 5, top - 5), (right + 5, bottom + 5), (0, 255, 0), 4,
                                  cv2.LINE_AA)

                    # 给68特征点识别取得一个转换顏色的frame
                    landmarks_frame = cv2.cvtColor(realTimeFrame, cv2.COLOR_BGR2RGB)

                    # 找出特征点位置, 参数为转换色彩空间后的帧图像以及脸部位置
                    shape = predictor_5(landmarks_frame, rect)

                    # 绘制5个特征点
                    for i in range(5):
                        cv2.circle(realTimeFrame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), 2)
                        cv2.putText(realTimeFrame, str(i), (shape.part(i).x, shape.part(i).y),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    (255, 0, 0), 1)

                    # 图像/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                    cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                                1)
                    cv2.putText(realTimeFrame, 'Person Count: {}/{}'.format(self.attendance_num, self.stu_num),
                                (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 0, 155), 2)

                    # 人脸识别
                    if self.isFaceRecognizerEnabled:
                        # 蓝色识别框（RGB三通道色参数其实顺序是BGR）
                        cv2.rectangle(realTimeFrame, (left, top), (right, bottom), (232, 138, 30), 2)
                        # 预测函数，识别后返回face ID和差异程度，差异程度越小越相似
                        face_features = facerec.compute_face_descriptor(frame, shape)
                        face_id, confidence = self.cal_best_match(face_features, all_stu_features)
                        logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                        if self.isDebugMode:  # 调试模式输出每帧识别信息
                            CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                        # 若置信度评分小于置信度阈值，认为是可靠识别
                        if confidence < 0.45:

                            # 从数据库中获取识别人脸的身份信息
                            try:
                                cursor.execute("SELECT * FROM users WHERE face_id=%s", (face_id,))
                                result = cursor.fetchall()
                                if result:
                                    stu_id = str(result[0][0])  # 学号
                                    zh_name = result[0][2]  # 中文名
                                else:
                                    raise Exception
                            except Exception as e:
                                logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                stu_id = ''
                                zh_name = ''
                                en_name = ''

                            isKnown = True

                            # 蓝色中文名标签
                            realTimeFrame = cv2ImgAddText(realTimeFrame, zh_name, left - 5, top - 10, (0, 97, 255))

                            if self.isPanalarmEnabled:  # 签到系统启动状态下执行
                                stu_statu = self.attendance_list.get(stu_id, 0)
                                if stu_statu > 7:
                                    realTimeFrame = cv2ImgAddText(realTimeFrame, '已识别', right - 45, top - 10,
                                                                  (0, 97, 255))  # 帧签到状态标记
                                elif stu_statu <= 6:
                                    # 连续帧识别判断，避免误识
                                    self.attendance_list[stu_id] = stu_statu + 1
                                else:
                                    attendance_time = datetime.now()
                                    self.attendance_list[stu_id] = stu_statu + 1
                                    alarmSignal = {
                                        'id': stu_id,
                                        'name': zh_name,
                                        'time': attendance_time,
                                        'img': realTimeFrame,
                                    }
                                    CoreUI.attendance_queue.put(alarmSignal)  # 签到队列插入该信号
                                    self.attendance_num += 1
                                    logging.info('系统发出了新的签到信号')
                            # 置信度标签
                            cv2.putText(realTimeFrame, str(round((1 - confidence) * 100, 4)),
                                        (left - 5, bottom + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 1)
                            # 蓝色英文名标签
                            cv2.putText(realTimeFrame, en_name, (left - 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 97, 255), 2)

                            know_faces.add(stu_id)  # 连续帧识别加入当前帧存在的人脸
                            if self.isDebugMode:  # 调试模式输出每帧识别信息
                                print(know_faces)
                                print(self.attendance_list)
                        else:
                            cv2.putText(realTimeFrame, str(round((1 - confidence) * 100, 3)),
                                        (left - 5, bottom + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 50, 255), 1)
                            # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                            cv2.putText(realTimeFrame, 'unknown', (left - 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2) def run(self):
        # 遇到的坑：因为FaceProcess在初始化的时候才认为准备running，而FaceProcess不running就不会有处理过的帧进入队列
        # 也就不会更新镜头信息帧到主界面，因此必须在打开摄像头时认为FaceProcess启动，改变stop时False状态
        self.isRunning = True
        # 帧数、人脸ID初始化
        frameCounter = 0
        currentFaceID = 0

        # 人脸跟踪器字典初始化
        faceTrackers = dict()
        all_stu_features = []

        isTrainingDataLoaded = False  # 预加载训练数据标记，检查一次过后即可不检查
        isDbConnected = False  # 预连接数据库标记，连接一次后即可只检查标记

        while self.isRunning and CoreUI.cap.isOpened():
            ret, frame = CoreUI.cap.read()  # 从摄像头捕获帧

            # 预加载识别数据
            if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):  # 训练数据
                recognizer = cv2.face.LBPHFaceRecognizer_create()  # LBPH人脸识别对象
                recognizer.read(CoreUI.trainingData)  # 读取训练数据
                if os.path.exists(CoreUI.dlib_features_data):
                    csv_reader = pd.read_csv(CoreUI.dlib_features_data, header=None)
                    all_stu_features = self.read_dlib_features_csv(csv_reader)

                isTrainingDataLoaded = True

            if not isDbConnected:  # 学生信息数据库
                conn, cursor = connect_to_sql()
                isDbConnected = True

            captureData = {}  # 单帧识别结果
            realTimeFrame = frame.copy()  # copy原始帧的识别帧
            alarmSignal = {}  # 报警信号

            # haar级联分类器+LBPH局部二值模式识别方法
            if self.is_haar_faceCascade:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度图，改变颜色空间，实际上就是BGR三色2(to) 灰度GRAY
                faces = self.find_faces_by_haar(frame)  # 分类器获取人脸

                # 人脸跟踪
                # Reference：https://github.com/gdiepen/face-recognition
                if self.isFaceTrackerEnabled:
                    know_faces = set()

                    # 删除质量较低的跟踪器
                    self.del_low_quality_face_tracker(faceTrackers, realTimeFrame)

                    # 遍历所有侦测到的人脸坐标
                    for (_x, _y, _w, _h) in faces:

                        # 微笑检测
                        smiles_y = (_y + _h + _y) // 2
                        smiles = haar_smile_cascade.detectMultiScale(gray[smiles_y:_y + _h, _x:_x + _w], 1.3, 10,
                                                                     minSize=(10, 10))
                        for (x, y, w, h) in smiles:
                            cv2.rectangle(realTimeFrame, (_x + x, smiles_y + y), (_x + x + w, smiles_y + y + h),
                                          (200, 50, 0), 1)
                            break

                        # 眼部检测
                        eyes = haar_eyes_cascade.detectMultiScale(gray[_y:smiles_y, _x:_x + _w], 1.3, 10,
                                                                  minSize=(40, 40))
                        for (x, y, w, h) in eyes:
                            cv2.rectangle(realTimeFrame, (_x + x, _y + y), (_x + x + w, _y + y + h), (150, 255, 30), 1)

                        isKnown = False

                        # 人脸识别
                        if self.isFaceRecognizerEnabled:
                            # 蓝色识别框（RGB三通道色参数其实顺序是BGR）
                            cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (232, 138, 30), 2)
                            # 预测函数，识别后返回face ID和差异程度，差异程度越小越相似
                            face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])
                            logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                            if self.isDebugMode:  # 调试模式输出每帧识别信息
                                CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                            # 从数据库中获取识别人脸的身份信息
                            try:
                                cursor.execute("SELECT * FROM users WHERE face_id=%s", (face_id,))
                                result = cursor.fetchall()
                                if result:
                                    stu_id = str(result[0][0])  # 学号
                                    zh_name = result[0][2]  # 中文名
                                    en_name = result[0][3]  # 英文名
                                else:
                                    raise Exception
                            except Exception as e:
                                logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                stu_id = ''
                                zh_name = ''
                                en_name = ''

                            # 若置信度评分小于置信度阈值，认为是可靠识别
                            if confidence < self.confidenceThreshold:
                                isKnown = True
                                if self.isPanalarmEnabled:  # 签到系统启动状态下执行
                                    stu_statu = self.attendance_list.get(stu_id, 0)
                                    if stu_statu > 9:
                                        realTimeFrame = cv2ImgAddText(realTimeFrame, '已识别', _x + _w - 45, _y - 10,
                                                                      (0, 97, 255))  # 帧签到状态标记
                                    elif stu_statu <= 8:
                                        # 连续帧识别判断，避免误识
                                        self.attendance_list[stu_id] = stu_statu + 1
                                    else:
                                        attendance_time = datetime.now()
                                        self.attendance_list[stu_id] = stu_statu + 1
                                        alarmSignal = {
                                            'id': stu_id,
                                            'name': zh_name,
                                            'time': attendance_time,
                                            'img': realTimeFrame
                                        }
                                        CoreUI.attendance_queue.put(alarmSignal)  # 签到队列插入该信号
                                        logging.info('系统发出了新的签到信号')
                                # 置信度标签
                                cv2.putText(realTimeFrame, str(round(100 - confidence, 3)), (_x - 5, _y + _h + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 255, 255), 1)
                                # 蓝色英文名标签
                                cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 97, 255), 2)
                                # 蓝色中文名标签
                                realTimeFrame = cv2ImgAddText(realTimeFrame, zh_name, _x - 5, _y - 10, (0, 97, 255))

                                know_faces.add(stu_id)
                                if self.isDebugMode:  # 调试模式输出每帧识别信息
                                    print(know_faces)
                                    print(self.attendance_list)
                            else:
                                cv2.putText(realTimeFrame, str(round(100 - confidence, 3)), (_x - 5, _y + _h + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 50, 255), 1)
                                # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                                cv2.putText(realTimeFrame, 'unknown', (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                                # 若置信度评分超出自动报警阈值，触发报警信号
                                if confidence > self.autoAlarmThreshold:
                                    # 报警系统是否开启
                                    if self.isPanalarmEnabled:  # 记录报警时间戳和当前帧
                                        alarmSignal['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')
                                        alarmSignal['img'] = realTimeFrame
                                        CoreUI.alarmQueue.put(alarmSignal)  # 报警队列插入该信号
                                        logging.info('系统发出了未知人脸信号')

                        # 帧数计数器
                        frameCounter += 1

                        # 每读取10帧，更新检测跟踪器的新增人脸
                        if frameCounter % 10 == 0:
                            frameCounter = 0  # 防止爆int
                            # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                            # 而dlib人脸跟踪器要求的是int类型
                            x, y, w, h = int(_x), int(_y), int(_w), int(_h)

                            # 计算中心点
                            x_bar = x + 0.5 * w
                            y_bar = y + 0.5 * h

                            # matchedFid表征当前检测到的人脸是否已被跟踪，未赋值则
                            matchedFid = None

                            # 将OpenCV中haar分类器获取的人脸位置与dlib人脸追踪器的位置做对比
                            # 上方坐标表示分类器检测结果，下方坐标表示遍历多目标追踪器检查有没有坐标上重合的脸，如果有，matchFid被赋值，说明该脸已追踪
                            # 如果没有，说明该分类器捕获的脸没有被追踪，那么多目标追踪器需要分配新的fid和追踪器实例

                            # 遍历人脸追踪器的face_id
                            for fid in faceTrackers.keys():
                                # 获取人脸跟踪器的位置
                                # tracked_position 是 dlib.drectangle 类型，用来表征图像的矩形区域，坐标是浮点数
                                tracked_position = faceTrackers[fid].get_position()
                                # 浮点数取整
                                t_x = int(tracked_position.left())
                                t_y = int(tracked_position.top())
                                t_w = int(tracked_position.width())
                                t_h = int(tracked_position.height())

                                # 计算人脸跟踪器的中心点
                                t_x_bar = t_x + 0.5 * t_w
                                t_y_bar = t_y + 0.5 * t_h

                                # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                                # 说明当前人脸已被跟踪
                                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                        (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                    matchedFid = fid

                            # 如果当前检测到的人脸是陌生人脸且未被跟踪
                            if not isKnown and matchedFid is None:
                                # 创建一个追踪器
                                tracker = dlib.correlation_tracker()  # 多目标追踪器
                                # 设置图片中被追踪物体的范围，也就是一个矩形框
                                tracker.start_track(realTimeFrame, dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                                # 将该人脸跟踪器分配给当前检测到的人脸
                                faceTrackers[currentFaceID] = tracker
                                # 人脸ID自增
                                currentFaceID += 1

                    # 遍历人脸跟踪器，输出追踪人脸的位置
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # 在跟踪帧中绘制方框圈出人脸，红框
                        cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                        # 图像/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                        cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1)
                    del_list = []
                    for stu_id, value in self.attendance_list.items():
                        if stu_id not in know_faces and value <= 8:
                            del_list.append(stu_id)
                    for stu_id in del_list:
                        self.attendance_list.pop(stu_id, 0)

            else:
                # dlib人脸关键点识别,绿框
                face_rects = self.find_faces_by_dlib(frame)
                # print(face_rects, scores, idx)  # rectangles[[(281, 209) (496, 424)]] [0.07766452427949444] [4]

                if self.isFaceTrackerEnabled:
                    know_faces = set()
                    # 删除质量较低的跟踪器
                    # self.del_low_quality_face_tracker(faceTrackers, realTimeFrame)

                    # 遍历所有侦测到的人脸坐标
                    for rect in face_rects:

                        isKnown = False

                        left = rect.left()
                        top = rect.top()
                        right = rect.right()
                        bottom = rect.bottom()

                        # 绘制出侦测人臉的矩形范围,绿框
                        cv2.rectangle(realTimeFrame, (left - 5, top - 5), (right + 5, bottom + 5), (0, 255, 0), 4,
                                      cv2.LINE_AA)

                        # 给68特征点识别取得一个转换顏色的frame
                        landmarks_frame = cv2.cvtColor(realTimeFrame, cv2.COLOR_BGR2RGB)

                        # 找出特征点位置, 参数为转换色彩空间后的帧图像以及脸部位置
                        shape = predictor_5(landmarks_frame, rect)

                        # 绘制5个特征点
                        for i in range(5):
                            cv2.circle(realTimeFrame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), 2)
                            cv2.putText(realTimeFrame, str(i), (shape.part(i).x, shape.part(i).y),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.5,
                                        (255, 0, 0), 1)

                        # 图像/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                        cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                                    1)
                        cv2.putText(realTimeFrame, 'Person Count: {}/{}'.format(self.attendance_num, self.stu_num),
                                    (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 0, 155), 2)

                        # 人脸识别
                        if self.isFaceRecognizerEnabled:
                            # 蓝色识别框（RGB三通道色参数其实顺序是BGR）
                            cv2.rectangle(realTimeFrame, (left, top), (right, bottom), (232, 138, 30), 2)
                            # 预测函数，识别后返回face ID和差异程度，差异程度越小越相似
                            face_features = facerec.compute_face_descriptor(frame, shape)
                            face_id, confidence = self.cal_best_match(face_features, all_stu_features)
                            logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                            if self.isDebugMode:  # 调试模式输出每帧识别信息
                                CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                                # 蓝色中文名标签
                                realTimeFrame = cv2ImgAddText(realTimeFrame, zh_name, left - 5, top - 10, (0, 97, 255))

                                if self.isPanalarmEnabled:  # 签到系统启动状态下执行
                                    stu_statu = self.attendance_list.get(stu_id, 0)
                                    if stu_statu > 7:
                                        realTimeFrame = cv2ImgAddText(realTimeFrame, '已识别', right - 45, top - 10,
                                                                      (0, 97, 255))  # 帧签到状态标记
                                    elif stu_statu <= 6:
                                        # 连续帧识别判断，避免误识
                                        self.attendance_list[stu_id] = stu_statu + 1
                                    else:
                                        attendance_time = datetime.now()
                                        self.attendance_list[stu_id] = stu_statu + 1
                                        alarmSignal = {
                                            'id': stu_id,
                                            'name': zh_name,
                                            'time': attendance_time,
                                            'img': realTimeFrame,
                                        }
                                        CoreUI.attendance_queue.put(alarmSignal)  # 签到队列插入该信号
                                        self.attendance_num += 1
                                        logging.info('系统发出了新的签到信号')
                                # 置信度标签
                                cv2.putText(realTimeFrame, str(round((1 - confidence) * 100, 4)),
                                            (left - 5, bottom + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 255, 255), 1)
                                # 蓝色英文名标签
                                cv2.putText(realTimeFrame, en_name, (left - 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 97, 255), 2)

                                know_faces.add(stu_id)  # 连续帧识别加入当前帧存在的人脸
                                if self.isDebugMode:  # 调试模式输出每帧识别信息
                                    print(know_faces)
                                    print(self.attendance_list)
                            else:
                                cv2.putText(realTimeFrame, str(round((1 - confidence) * 100, 3)),
                                            (left - 5, bottom + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 50, 255), 1)
                                # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                                cv2.putText(realTimeFrame, 'unknown', (left - 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)

                    del_list = []
                    for stu_id, value in self.attendance_list.items():
                        if stu_id not in know_faces and value <= 6:
                            del_list.append(stu_id)
                    for stu_id in del_list:
                        self.attendance_list.pop(stu_id, 0)
                captureData['originFrame'] = frame
                captureData['realTimeFrame'] = realTimeFrame
                CoreUI.captureQueue.put(captureData)

                # 停止OpenCV线程

            def stop(self):
                self.isRunning = False
                self.quit()
                self.wait()