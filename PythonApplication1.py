import numpy as np
import math
import tkinter as tk
import cv2
import cv2.aruco as aruco
from PIL import Image, ImageTk
import time
import serial

# bot number is 2
window = tk.Tk()
window.iconbitmap(default="favicon.ico")
window.title("visual servoing")
window.minsize(640,600)
class window_tk():
    def __init__(self,main):
        self.canvas = tk.Canvas(main, bg='white', height=480, width=640 )
        # constant values
        #--------------------
        # calibiration values
        self.pitocm = 0.0176 # 1 px to cm in 144dpi screen
        self.camHeight = 480 # height in px
        self.camWidth = 640  # width in px
        self.worldheight = 126 # height convered by cam at a height of 25 cm is 126cm
        self.worldWidth = 163  # width covered by cam at a width of 25 cm is 163 cm 
       
        self.camHeightCm = self.camHeight*self.pitocm
        self.camWidthCm = self.camWidth*self.pitocm
        self.isReadyNxt = True
        self.img = None
        self.frame = None
        self.corners = None
        self.theta = None
        self.bg = self.canvas.create_image(0,0,anchor = tk.NW,image=self.img)
        self.vid = None
        self.ids = [5,7,10]
        self.indexId = {}
        self.ser = serial.Serial(port="COM3",baudrate=115200,bytesize=8,timeout=2,stopbits=serial.STOPBITS_ONE)
        self.oval = None
        self.destination = {} # save the final point for travesring also initialize how many bot arre there
        self.odomentary = {}
        self.side = tk.Frame()
        self.but1 = tk.Button(self.side,text="start")
        self.but2 = tk.Button(self.side,text="stop")
        self.but3 = tk.Button(self.side,text="Bot A",command=lambda:self.botDestination(1,"red"))
        self.but4 = tk.Button(self.side,text="Bot B",command=lambda:self.botDestination(2,"green"))
        
    def botDestination(self,bot,color):
        self.canvas.bind('<Motion>',self.motion)
        self.canvas.bind('<Button-1>',lambda event,bot=bot,color=color: self.drawDestination(event,bot,color))
        print("bot number",bot)
 
    def motion(self,event):
        x,y = event.x,event.y
        self.canvas.coords(self.oval,x-10,y-10,x+10,y+10)

    def drawDestination(self,event,bot,color):
        x,y = event.x,event.y
        if(bot == 1):
            self.destination[bot]= [x,y]
        elif(bot == 2):
            self.destination[bot] = [x,y]
        try:
            self.canvas.delete(color)
        except:
            print("nothing to delete")
        self.canvas.create_oval(x-10,y-10,x+10,y+10,fill=color,outline="white",tags=color)
        self.canvas.unbind('<Button-1>')
        print("Destination",self.destination)
    def activate_video(self):
        self.vid = cv2.VideoCapture(1)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.canvas.pack(anchor= tk.NW)
        self.oval = self.canvas.create_oval(0,0,0,0,fill="grey",outline="white")
        self.side.pack(anchor=tk.NW,fill='both')
        self.but1.pack(side=tk.LEFT,ipadx=20,ipady=10)
        self.but2.pack(side=tk.LEFT,ipadx=20,ipady=10)
        self.but3.pack(side=tk.LEFT,ipadx=20,ipady=10)
        self.but4.pack(side=tk.LEFT,ipadx=20,ipady=10)
        self.show_frames()

    def show_frames(self):
        ret,self.frame = self.vid.read()
        self.frame= cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        corners,ids = self.detect_aruco(self.frame)
        if len(corners) != 0:
            temp = np.array(ids)
            temp = temp.tolist()
            self.idtobotmatcher(temp)
            self.corners  = np.array(corners)
            cali = self.getMarkerCenter(self.corners[0][0])
            print("calibration",cali)
            aruco.drawDetectedMarkers(self.frame, corners)
            if(len(self.destination)>0):
                for i in range(len(self.destination)):
                    try:
                        self.odomentaryData(self.corners[self.indexId[i+1]][0],self.destination[i+1],i+1)
                        self.botSteer(i+1)
                    except:
                        continue
        if not ret:
            print("can't receive the end of the frame")
            exit
        img = Image.fromarray(self.frame)
        self.img = ImageTk.PhotoImage(image = img)
        self.canvas.itemconfig(self.bg,image=self.img)
        self.canvas.after(100,self.show_frames)

    def sendCommands(self,cmd):
        print("cmds--.",cmd)
        self.ser.write(cmd.encode())
        print("cmds sent")
        receive = self.ser.readline()
        time.sleep(1)
        print(receive.decode())
    def detect_aruco(self,frame):
        # dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        # parameters =  aruco.DetectorParameters()
        # detector = aruco.ArucoDetector(dictionary, parameters)
        # old cv2 version aruc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        arucoParams = aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
        # corners, ids, rejected = detector.detectMarkers(frame)
        return corners,ids
    def anglerotated(self,corners,dest):
        print("inside it")
        unit_x_axis = [1.,0.]
        dest = np.array(dest)
        center = self.getMarkerCenter(corners)
        top_edge_midpoint = (corners[0]+corners[1])/2.
        unit_vec = (top_edge_midpoint-center)/np.linalg.norm(top_edge_midpoint-center) 
        dest_unit_vec = (dest-center)/np.linalg.norm(dest-center) 
        dot = unit_vec[0]*unit_x_axis[0] + unit_vec[1]*unit_x_axis[1] 
        det = unit_vec[0]*unit_x_axis[1] - unit_vec[1]*unit_x_axis[0]
        theta = math.atan2(det,dot)
        theta = round(np.rad2deg(theta),0)
        dot1 = dest_unit_vec[0]*unit_x_axis[0] + dest_unit_vec[1]*unit_x_axis[1] 
        det1 = dest_unit_vec[0]*unit_x_axis[1] - dest_unit_vec[1]*unit_x_axis[0]
        dest_angle = math.atan2(det1,dot1)
        dest_angle = round(np.rad2deg(dest_angle),0)
        print("bot angle - {} , dest to bot angle {}".format(theta,dest_angle))
        return theta,dest_angle

    def getMarkerCenter(self,corners):
        px = (corners[0][0] + corners[1][0] + corners[2][0]+ corners[3][0]) * 0.25
        py = (corners[0][1] + corners[1][1] + corners[2][1]+ corners[3][1]) * 0.25
        return [px,py]
    def odomentaryData(self,corners,bot,botID):
        if len(bot)>1: # only calling this function if bot destination is assigned 
            print("inside ododmentary function")
            center = self.getMarkerCenter(corners)
            center_vec_mag = np.linalg.norm(np.array(bot)-np.array(center))
            left_vec_mag = np.linalg.norm(np.array(bot)-np.array(corners[0]))
            right_vec_mag = np.linalg.norm(np.array(bot)-np.array(corners[1]))
            theta,crcted_angle = self.anglerotated(corners,bot)
            self.odomentary[botID]=[left_vec_mag,center_vec_mag,right_vec_mag,theta,crcted_angle] 
            print("{}-id {}".format(botID,self.odomentary[botID]))
            cv2.line(self.frame,(int(bot[0]),int(bot[1])),(int(center[0]),int(center[1])),(0,255,0),1)
            cv2.line(self.frame,(int(bot[0]),int(bot[1])),(int(corners[0][0]),int(corners[0][1])),(255,0,0),1)
            cv2.line(self.frame,(int(bot[0]),int(bot[1])),(int(corners[1][0]),int(corners[1][1])),(0,0,255),1)
    def idtobotmatcher(self,ids):
        print("ids",ids)
        count = 0
        for i in self.ids:
            if(ids.count([i]) and count+1 not in self.indexId):
                self.indexId[count+1] = ids.index([i])
                count+=1
        print(self.indexId)
    def destinationMapper(self,botId):
        dest = self.destination[botId]
        xCm = dest[0]*self.pitocm
        yCm = dest[1]*self.pitocm
        wX = (self.worldWidth/self.camWidthCm) * xCm
        wY = (self.worldHeight/self.camHeightCm) * yCm
    def steeringBehaviour(self,botId):
        odomentary = self.odomentary[botId]
        Dl = odomentary[0]
        Dr = odomentary[2]
        Dc = odomentary[1]
        omega = odomentary[3]
        if(Dl>Dr):
            pass
        elif(Dl<Dr):
            pass
        elif(Dl==Dr):
            pass
        
    def botSteer(self,botId):
        odo = self.odomentary[botId]
        Dl = odo[0]
        Dc = odo[1]
        Dr = odo[2]
        theta = odo[3]
        dist_theta = odo[4]
        if(Dl>Dr and abs(Dl-Dr)>20): #right
            print("inside 1st if")
            angle = abs(theta - dist_theta)
            print(angle)
            ticks = angle%15
            ticks = int(ticks*50)
            cmd = "<4 "+str(ticks)+">"
            print(cmd)
            self.sendCommands(cmd)
        elif(Dl<Dr and abs(Dl-Dr)>20): # left
            angle = abs(theta - dist_theta)
            print(angle)
            ticks = angle%15
            ticks = int(ticks*50)
            cmd = "<3 "+str(ticks)+">"
            print(cmd)
            self.sendCommands(cmd)
        # elif(abs(Dl-Dr)<=10): # straight
        #     angle = abs(theta - dist_theta)
        #     print(angle)
        #     ticks = angle % 15
        #     ticks = int(ticks * 50)
        #     cmd = "<1 500>"
        #     print(cmd)
        #     self.sendCommands(cmd)

    #def isReady(self,interval,currentTime=round(time.time()*1000),prevTime=round(time.time()*1000)):
    #    while((currentTime - prevTime) < interval):
    #        currentTime = round(time.time()*1000)
    #    return True
    
GUI = window_tk(window)
GUI.activate_video()
window.mainloop()