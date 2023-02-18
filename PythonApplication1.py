import numpy as np
import math
import tkinter as tk
import cv2
import cv2.aruco as aruco
from PIL import Image, ImageTk
import time
import serial
from warnings import warn
import heapq
from shapely.geometry import box

# tkinter
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

        # for cuda constants
        # threshold values that will be used to identify the useful bounding boxes
        self.SCORE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.CONFIDENCE_THRESHOLD = 0.4
        self.is_cuda = True
        self.net = None


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
        self.net = self.build_model(self.is_cuda)
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

    def format_yolov5(self,frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self,image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.camWidth, self.camWidth), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds

    def unwraping(self,img,out_data):

        img_width, img_height, _ = img.shape
        x_factor = img_width / self.camWidth
        y_factor = img_height / self.camWidth

        class_ids = []
        confidences = []
        boxes = []
        points = []

        rows = out_data.shape[0]

        for r in range(rows):
            row = out_data[r]
            conf = row[4]
            if conf >= 0.4:
                class_scores = row[5:]
                _, _, _, max_ind = cv2.minMaxLoc(class_scores)
                class_id = max_ind[1]
                if (class_scores[class_id] > 0.25):
                    confidences.append(conf)
                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    new_x = x * x_factor
                    new_y = y * y_factor
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    point = np.array([new_x, new_y])
                    points.append(point)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        result_confidences = []
        result_class_ids = []
        result_boxes = []
        result_points = []
        for ind in indices:
            result_confidences.append(confidences[ind])
            result_class_ids.append(class_ids[ind])
            result_boxes.append(boxes[ind])
            result_points.append(points[ind])

        return result_class_ids, result_confidences, result_boxes, result_points

    def list_dim(self,ids, points, boxes):
        white_list = []
        square_point = []
        # return a list with tuples inside [class_id,(x,y),(width,height),square_side]
        for (id, point, box) in zip(ids, points, boxes):
            limit = 100
            sq_l = 120
            if id == 1:
                # sq_l = box[2]
                if box[2] > limit:
                    sq_l = box[2]
                    point = (point[0], point[1])
                    square_point.append(point)  # square_point= [point[0],point[1]]
                    # print(box)
            elif id == 2:
                white_list.append(id)
                point = (point[0], point[1])
                white_list.append(point)
                box = (box[2], box[3])
                white_list.append(box)
                # white_list.append(sq_l)
        return white_list, square_point, sq_l

    # the grid is constructed by avoiding obstacles i.e., the center pixel values of the obstacles is replaced by [0,0]
    def allgrid(self, white_list, sq_list, sq_l):
        white_x = white_list[1][0]  # 345.7974548339844
        white_y = white_list[1][1]  # 251.01719665527344
        w = white_list[2][0]  # 420
        # row = round(w/val)
        h = white_list[2][1]  # 315
        new_x = int(white_x - 0.5 * w)
        new_y = int(white_y - 0.5 * h)
        # w = w + new_x
        # h = h + new_y
        centers = []
        path_pt = []
        step_size = sq_l
        row = round(w / step_size)
        col = round(h / step_size)
        w = step_size * row
        h = step_size * col
        for y in range(new_y, h, step_size):
            for x in range(new_x, w, step_size):
                centers.append([(x + (step_size * 0.5)), (y + (step_size * 0.5))])
        grid_pts = centers
        # print(centers)
        final_path = []
        new_list = sq_list

        for grid_pt in grid_pts:
            for sq_cen in new_list:
                if math.dist(sq_cen, grid_pt) < (step_size * 0.5):  # the grid_pt coincides with obstacle
                    i = centers.index(grid_pt)
                    centers[i] = [0, grid_pt]

        for center in centers:
            if center[0] == 0:
                final_path.append(center)
            else:
                final_path.append([1, center])
        return final_path

    def createarray(self,path_centers):
        value_array = np.empty(shape=(3, 5), dtype=object)
        full_view = np.empty(shape=(3, 5), dtype=object)
        row, col = value_array.shape

        l = 0
        for x in range(row):
            for y in range(col):
                if path_centers[l][0] == 1:
                    value_array[x][y] = path_centers[l]  # path center exists i.e., 1
                    full_view[x][y] = 1
                else:
                    value_array[x][y] = path_centers[l]  # path_center does not exist it is an obstacle
                    full_view[x][y] = 0
                l += 1
        return value_array, full_view  # value_array gives the center and full_view gives the binary representation

    def show_frames(self):
        ret,self.frame = self.vid.read()
        inputImage = self.format_yolov5(self.frame)
        outs = self.detect(inputImage)
        class_ids, confidences, boxes, points = self.unwraping(inputImage, outs[0])
        array_of_center = None
        try:
            white_list, square_point, sq_l = self.list_dim(class_ids, points,boxes)  # sq_l is the pix length of each obstacle
            # cv2.circle(self.frame,(int(square_point[0][0]),int(square_point[0][1])),10,(255,0,0),-1)
            path_centers = self.allgrid(white_list, square_point, sq_l)
            array_of_center, binary_val = self.createarray(path_centers)
            maze = binary_val.tolist()
            start= (0,0)
            end = (2,0)
            path = self.astar(maze, start, end)
            # print(path)
        except:
            print("error in detection")
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
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
                        destIndex = self.findindex(self.destination[i+1],array_of_center)
                        print("destination index",destIndex)
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
    def findindex(self,dest,arr):
        row,col = np.shape(arr)
        print("inside destination")
        index = None
        if arr is not None:
            for r in range(row):
                for c in range(col):
                    if math.dist(arr[r][c][1],dest)<=50:
                        index = [r,c]
        return index
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
        # print("bot angle - {} , dest to bot angle {}".format(theta,dest_angle))
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
            # print("{}-id {}".format(botID,self.odomentary[botID]))
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
        # print(self.indexId)

    def botSteer(self,botId):
        odo = self.odomentary[botId]
        Dl = odo[0]
        Dc = odo[1]
        Dr = odo[2]
        theta = odo[3]
        dist_theta = odo[4]
        if(Dc >= 25):
            if(Dl>Dr and abs(Dl-Dr)>20): #right
                # angle = abs(theta - dist_theta)
                # print(angle)
                # ticks = angle%15
                ticks = 200 #int(ticks*50)
                cmd = "<4 "+str(ticks)+">"
                print(cmd)
                self.sendCommands(cmd)
            elif(Dl<Dr and abs(Dl-Dr)>20): # left
                # angle = abs(theta - dist_theta)
                # print(angle)
                # ticks = angle%15
                ticks =200    #int(ticks*50)
                cmd = "<3 "+str(ticks)+">"
                print(cmd)
                self.sendCommands(cmd)
            elif(abs(Dl-Dr)<=20): # straight
                ticks = int(200)
                cmd = "< 1 "+ str(ticks)+ ">"
                print(cmd)
                self.sendCommands(cmd)

    def build_model(self,is_cuda):
        net = cv2.dnn.readNet("first_bot.onnx")
        if is_cuda:
            print("Attempt to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    def return_path(self,current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Return reversed path

    def astar(self,maze, start, end):
        """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param start:
        :param end:
        :return:
        """

        # Create start and end node
        start_node = self.Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = self.Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Heapify the open_list and Add the start node
        heapq.heapify(open_list)
        heapq.heappush(open_list, start_node)

        # Adding a stop condition
        outer_iterations = 0
        max_iterations = (len(maze[0]) * len(maze) // 2) * 5

        # what squares do we search
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)

        # Loop until you find the end
        while len(open_list) > 0:
            outer_iterations += 1

            if outer_iterations > max_iterations:
                # if we hit this point return the path such as it is
                # it will not contain the destination
                warn("giving up on pathfinding too many iterations")
                return self.return_path(current_node)

                # Get the current node
            current_node = heapq.heappop(open_list)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                return self.return_path(current_node)

            # Generate children
            children = []

            for new_position in adjacent_squares:  # Adjacent squares

                # Get node position
                node_position = (
                current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 1:
                    continue

                # Create new node
                new_node = self.Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                if len([open_node for open_node in open_list if
                        child.position == open_node.position and child.g > open_node.g]) > 0:
                    continue

                # Add the child to the open list
                heapq.heappush(open_list, child)

        warn("Couldn't get a path to destination")
        return None
    class Node:
        """
        A node class for A* Pathfinding
        """

        def __init__(self, parent=None, position=None):
            self.parent = parent
            self.position = position

            self.g = 0  # G is the distance between the current node and the start node.
            self.h = 0  # H is the heuristic — estimated distance from the current node to the end node.
            self.f = 0  # F is the total cost of the node.

        def __eq__(self, other):
            return self.position == other.position

        def __repr__(self):
            return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

        # defining less than for purposes of heap queue
        def __lt__(self, other):
            return self.f < other.f

        # defining greater than for purposes of heap queue
        def __gt__(self, other):
            return self.f > other.f
    
GUI = window_tk(window)
GUI.activate_video()
window.mainloop()