# -*- coding: utf-8 -*-

from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import sys
import threading
import cv2, imutils
import numpy as np
import time

from ui.ui_dash import *  # import gui


# # global vehicle1
# vehicle1 = 0
# # global vehicle2
# vehicle2 = 0
# # global vehicle3
# vehicle3 = 0
# # global vehicle4
# vehicle4 = 0

l_one_time=0
l_two_time=0
l_three_time=0
l_four_time=0


first_count=0
sec_count=0
third_count=0
foth_count =0



class main_window(QMainWindow):
    def __init__(self,vehicle1=0,vehicle2=0,vehicle3=0,vehicle4=0):



        self.vehicle1 = vehicle1
        self.vehicle2 = vehicle2
        self.vehicle3 = vehicle3
        self.vehicle4 = vehicle4

        self.l_one_time = l_one_time
        self.l_two_time = l_two_time
        self.l_three_time = l_three_time
        self.l_four_time = l_four_time


        self.first_count=first_count
        self.sec_count=sec_count
        self.third_count=third_count
        self.foth_count=foth_count

        # self.priodic = dict()
        # self.prio_tup = tuple()

        QMainWindow.__init__(self)


        self.main_ui = Ui_MainWindow()  # initilize gui
        self.main_ui.setupUi(self)  # initilize gui

        # thred

        self.thred1 = threading.Thread(target=self.lane1)
        self.thred2 = threading.Thread(target=self.lane2)
        self.thred3 = threading.Thread(target=self.lane3)
        self.thred4 = threading.Thread(target=self.lane4)

        # self.main_ui.btn1.clicked.connect(self.start_thred)

        self.start_thred()


    def start_thred(self):

        self.thred1.start()
        self.thred2.start()
        self.thred3.start()
        self.thred4.start()

    def setPhoto_lane1(self, image):

        self.tmp = image
        image = imutils.resize(image, width=450)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.main_ui.lbl1.setPixmap(QtGui.QPixmap.fromImage(image))

    def setPhoto_lane2(self, image):

        self.tmp = image
        image = imutils.resize(image, width=450)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.main_ui.lbl2.setPixmap(QtGui.QPixmap.fromImage(image))

    def setPhoto_lane3(self, image):

        self.tmp = image
        image = imutils.resize(image, width=450)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.main_ui.lbl3.setPixmap(QtGui.QPixmap.fromImage(image))

    def setPhoto_lane4(self, image):

        self.tmp = image
        image = imutils.resize(image, width=450)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.main_ui.lbl4.setPixmap(QtGui.QPixmap.fromImage(image))

    def lane1(self):
        # self.vehicle1=vehicle1
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolo3-spp.cfg')

        # classes = []
        with open('coco.names', 'r', ) as f:
            classes = f.read().splitlines()
        # method_1
        cap = cv2.VideoCapture('D:\Downloads\opencv\overpass.mp4')
        # cap.set(cv2.CAP_PROP_FPS,10)

        # cap = cv2.iread('office.jpg')
        # global vehicle1
        # vehicle1 = 0

        while (cap.isOpened()):
            ret, frame = cap.read()

            ret, frame1 = cap.read()
            frame2 = frame

            # extract foreground mask
            fgMask = cv2.absdiff(frame1, frame2)
            #fgMask = cv2.imread('fgMask.jpg',0)
            if not ret:
                return
            else:
                fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            # apply the threshold for increasing white foreground
            _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

            # assign frame2 to frame1 ro continue the iteration untill all frames are read
            frame1 = frame2

            # draw the reference line
            cv2.line(frame, (220, 370), (1000, 370), (0, 0, 255), 2)

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 480), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)

            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    xMid = int((x + (x + w)) / 2)
                    yMid = int((y + (y + w)) / 2)
                    cv2.circle(frame, (xMid, yMid), 5, (0, 0, 255), 5)
                    if yMid > 350 and y < 370 and label != "person":
                        self.vehicle1 += 1
                    cv2.putText(frame, label + "" + confidence, (x, y + 20), font, 2, (0, 0, 0), 2)

            cv2.putText(frame, 'total vehicles : {}'.format(self.vehicle1), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255),
                        2)

            # cv2.imshow('image', frame)

            self.setPhoto_lane1(frame)

            # cv2.imshow('fgmask', thresh)
            # cv2.imshow('frame1',frame1)
            # cv2.imshow('frame2',frame2)

            key = cv2.waitKey(1)
            if key == 100:
                break
        return self.vehicle1
        cap.release()
        cv2.destroyAllWindows()

    def lane2(self):
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolo3-spp.cfg')

        # classes = []
        with open('coco.names', 'r', ) as f:
            classes = f.read().splitlines()
        # method_1
        cap = cv2.VideoCapture('D:\Downloads\opencv\overpass.mp4')
        # cap.set(cv2.CAP_PROP_FPS,10)

        # cap = cv2.iread('office.jpg')
        # global vehicle2
        # vehicle2 = 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            ret, frame1 = cap.read()
            frame2 = frame

            # extract foreground mask
            fgMask = cv2.absdiff(frame1, frame2)
            #fgMask = cv2.imread('fgMask.jpg',0)
            #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            if not ret:

                return
            else:
                fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            # apply the threshold for increasing white foreground
            _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

            # assign frame2 to frame1 ro continue the iteration untill all frames are read
            frame1 = frame2

            # draw the reference line
            cv2.line(frame, (220, 370), (1000, 370), (0, 0, 255), 2)

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 480), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)

            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    xMid = int((x + (x + w)) / 2)
                    yMid = int((y + (y + w)) / 2)
                    cv2.circle(frame, (xMid, yMid), 5, (0, 0, 255), 5)
                    if yMid > 350 and y < 370 and label != "person":
                        self.vehicle2 += 1
                    cv2.putText(frame, label + "" + confidence, (x, y + 20), font, 2, (0, 0, 0), 2)

            cv2.putText(frame, 'total vehicles : {}'.format(self.vehicle2), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255),
                        2)

            # cv2.imshow('image', frame)

            self.setPhoto_lane2(frame)

            # cv2.imshow('fgmask', thresh)
            # cv2.imshow('frame1',frame1)
            # cv2.imshow('frame2',frame2)

            key = cv2.waitKey(1)
            if key == 100:
                break
        return self.vehicle2
        cap.release()
        cv2.destroyAllWindows()

    def lane3(self):
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolo3-spp.cfg')

        # classes = []
        with open('coco.names', 'r', ) as f:
            classes = f.read().splitlines()
        # method_1
        cap = cv2.VideoCapture('D:\Downloads\opencv\overpass.mp4')
        # cap.set(cv2.CAP_PROP_FPS,10)

        # cap = cv2.iread('office.jpg')
        # global vehicle3
        # vehicle3 = 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            ret, frame1 = cap.read()
            frame2 = frame

            # extract foreground mask
            fgMask = cv2.absdiff(frame1, frame2)
            #fgMask = cv2.imread('fgMask',0)
            #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            if not ret:

                return
            else:
                fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            # apply the threshold for increasing white foreground
            _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

            # assign frame2 to frame1 ro continue the iteration untill all frames are read
            frame1 = frame2

            # draw the reference line
            cv2.line(frame, (220, 370), (1000, 370), (0, 0, 255), 2)

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 480), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)

            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    xMid = int((x + (x + w)) / 2)
                    yMid = int((y + (y + w)) / 2)
                    cv2.circle(frame, (xMid, yMid), 5, (0, 0, 255), 5)
                    if yMid > 350 and y < 370 and label != "person":
                        self.vehicle3 += 1
                    cv2.putText(frame, label + "" + confidence, (x, y + 20), font, 2, (0, 0, 0), 2)

            cv2.putText(frame, 'total vehicles : {}'.format(self.vehicle3), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255),
                        2)

            # cv2.imshow('image', frame)

            self.setPhoto_lane3(frame)

            # cv2.imshow('fgmask', thresh)
            # cv2.imshow('frame1',frame1)
            # cv2.imshow('frame2',frame2)

            key = cv2.waitKey(1)
            if key == 100:
                break

        return self.vehicle3
        cap.release()
        cv2.destroyAllWindows()

    def lane4(self):
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolo3-spp.cfg')

        # classes = []
        with open('coco.names', 'r', ) as f:
            classes = f.read().splitlines()
        # method_1
        cap = cv2.VideoCapture('D:\Downloads\opencv\overpass.mp4')
        # cap.set(cv2.CAP_PROP_FPS,10)

        # cap = cv2.iread('office.jpg')
        # global vehicle4
        # vehicle4 = 0


        while (cap.isOpened()):
            ret, frame = cap.read()
            ret, frame1 = cap.read()
            frame2 = frame

            # extract foreground mask
            fgMask = cv2.absdiff(frame1, frame2)
            #fgMask = cv2.imread('fgMask.jpg',0)
            #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)
            if not ret:

                return
            else:
                fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)

            # apply the threshold for increasing white foreground
            _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

            # assign frame2 to frame1 ro continue the iteration untill all frames are read
            frame1 = frame2

            # draw the reference line
            cv2.line(frame, (220, 370), (1000, 370), (0, 0, 255), 2)

            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 480), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)

            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    xMid = int((x + (x + w)) / 2)
                    yMid = int((y + (y + w)) / 2)
                    cv2.circle(frame, (xMid, yMid), 5, (0, 0, 255), 5)
                    if yMid > 350 and y < 370 and label != "person":
                        self.vehicle4 += 1
                    cv2.putText(frame, label + "" + confidence, (x, y + 20), font, 2, (0, 0, 0), 2)

            cv2.putText(frame, 'total vehicles : {}'.format(self.vehicle4), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255),
                        2)

            # cv2.imshow('image', frame)

            self.setPhoto_lane4(frame)

            # cv2.imshow('fgmask', thresh)
            # cv2.imshow('frame1',frame1)
            # cv2.imshow('frame2',frame2)

            key = cv2.waitKey(1)
            if key == 100:
                break
        return self.vehicle4
        cap.release()
        cv2.destroyAllWindows()

    def lights(self):
        global priodic
        priodic = {}
        if self.vehicle1 != 0:
            priodic['vehicle1'] = self.vehicle1
            l_one_time = 20
        elif self.vehicle2 != 0:
            priodic['vehicle2'] = self.vehicle2
            l_two_time = 20
        elif self.vehicle3 != 0:
            priodic['vehicle3'] = self.vehicle3
            l_three_time = 20
        elif self.vehicle4 != 0:
            priodic['vehicle4'] = self.vehicle4
            l_four_time = 20



        prio_tup = tuple(sorted(priodic, reverse=True))
        # first_count = prio_tup[1]
        # sec_count = prio_tup[2]
        # third_count = prio_tup[3]
        # foth_count = prio_tup[4]

        # -----------------------------
        while len(prio_tup) != 0:

            if prio_tup[0] == 'vehicle1':

                #lane2,3,4 red >> on
                print('YES1')
                self.main_ui.lane2_r.setStyleSheet(u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane3_r.setStyleSheet(u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane4_r.setStyleSheet(u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")

                # red yello >> on
                self.main_ui.lane1_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)
                self.main_ui.lane1_y.setStyleSheet(
                    u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)

                # red yello >> off
                self.main_ui.lane1_r.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane1_y.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                #green >> on
                self.main_ui.lane1_g.setStyleSheet(
                    u"\n""background-color: rgb(0, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(0.5)


                #lane2,3,4 yello >> on
                if prio_tup[1] == 'vehicle2':
                    self.main_ui.lane2_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle3':
                    self.main_ui.lane3_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle4':
                    self.main_ui.lane4_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")

                time.sleep(l_one_time)
                # green >> off
                self.main_ui.lane1_g.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")
                prio_tup = (prio_tup[1:])


            elif prio_tup[0] == 'vehicle2':
                # lane1,3,4 red >> on
                self.main_ui.lane1_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane3_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane4_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")

                # red yello >> on
                self.main_ui.lane2_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)
                self.main_ui.lane2_y.setStyleSheet(
                    u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)

                # red yello >> off
                self.main_ui.lane2_r.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane2_y.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                # green >> on
                self.main_ui.lane2_g.setStyleSheet(
                    u"\n""background-color: rgb(0, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(0.5)

                # lane1,3,4 yello >> on
                if prio_tup[1] == 'vehicle1':
                    self.main_ui.lane1_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle3':
                    self.main_ui.lane3_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle4':
                    self.main_ui.lane4_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")

                time.sleep(l_two_time)
                # green >> off
                self.main_ui.lane2_g.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                prio_tup = prio_tup[1:]


            elif prio_tup[0] == 'vehicle3':
                # lane1,2,4 red >> on
                self.main_ui.lane1_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane2_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane4_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")

                # red yello >> on
                self.main_ui.lane3_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)
                self.main_ui.lane3_y.setStyleSheet(
                    u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)

                # red yello >> off
                self.main_ui.lane3_r.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane3_y.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                # green >> on
                self.main_ui.lane3_g.setStyleSheet(
                    u"\n""background-color: rgb(0, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(0.5)

                # lane1,2,4 yello >> on
                if prio_tup[1] == 'vehicle1':
                    self.main_ui.lane1_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle2':
                    self.main_ui.lane2_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle4':
                    self.main_ui.lane4_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")

                time.sleep(l_three_time)
                # green >> off
                self.main_ui.lane3_g.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                prio_tup = prio_tup[1:]

            elif prio_tup[0] == 'vehicle4':
                # lane1,2,3 red >> on
                self.main_ui.lane1_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane2_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane3_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")

                # red yello >> on
                self.main_ui.lane4_r.setStyleSheet(
                    u"\n""background-color: rgb(255, 0, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)
                self.main_ui.lane4_y.setStyleSheet(
                    u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(1.5)

                # red yello >> off
                self.main_ui.lane4_r.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")
                self.main_ui.lane4_y.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                # green >> on
                self.main_ui.lane4_g.setStyleSheet(
                    u"\n""background-color: rgb(0, 255, 0);\n""border-radius:24px;\n""\n""")
                time.sleep(0.5)

                # lane1,2,3 yello >> on
                if prio_tup[1] == 'vehicle1':
                    self.main_ui.lane1_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle2':
                    self.main_ui.lane2_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")
                elif prio_tup[1] == 'vehicle3':
                    self.main_ui.lane3_y.setStyleSheet(
                        u"\n""background-color: rgb(255, 255, 0);\n""border-radius:24px;\n""\n""")

                time.sleep(l_four_time)
                # green >> off
                self.main_ui.lane4_g.setStyleSheet(
                    u"\n""background-color: rgba(0, 0, 0,0.2);\n""border-radius:24px;\n""\n""")

                prio_tup = prio_tup[1:]

if __name__ == '__main__':
    app = QApplication(sys.argv)  # create app
    window = main_window()
    window.show()  # display app
    window.lights()
    sys.exit(app.exec_())
