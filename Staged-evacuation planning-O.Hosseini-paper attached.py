# coding=utf-8
# Python code developed by Omid Hosseini*, Mojtaba Maghrebi, Mahmood F. Maghrebi in the paper:
# Article : Automatically Determining Staged-Evacuation Plan to Minimize Total Evacuation Time, Congestion Severity and Fire Threats
# (0098)9150098089 - (0098)5138805101 - omid.s.hosseini@gmail.com - Civil engeneering dep, Ferdowsi university of Mashhad, Iran
# Published in the "  journal"
# DOI:
# Github repositionary:
# Please cite us if using this code.
# Note: The results that presented in the paper might have some differences while running the code below, because of testing variouse evacuation scenarios and various facility plan.


import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import pygame.draw
import copy
import multiprocessing

#Automatically Determining Staged-Evacuation Plan to Minimize Total Evacuation Time, Congestion Severity and Fire Threat

#Simulation parameters
proc_no=8
np.seterr(divide='ignore', invalid='ignore')
SCREENSIZE = [1000, 500]
RESOLUTION = 180
AGENTSNUM = 50
Num_Agents = AGENTSNUM
BACKGROUNDCOLOR = (255,255,255)
AGENTCOLOR = (255,0,0)
LINECOLOR = (0,0,255)
AGENTSIZE = [6,18]
AGENTSICKNESS = 1
FORCELINETICKNESS=1
WALLTICKNESS=1
FORCELINECOLOR1=[0,0,0]
FORCELINECOLOR2=(0,0,255)
ZOOMFACTOR = 4
num_of_zones=5
hall_lengh = #INPUT the specifications
hall_width = #INPUT the specifications
door_width = #INPUT the specifications
inner_door_width=#INPUT the specifications
door_pass_rate = 1
desired_speed=2
fire_speed = 0.35
Afactor = 1
Bfactor = 1
Dfactor = 1
tou = 0.1

#optimization parameters
pop_size = 32
max_gen = 31
min_x = 0
max_x =121
Step = 10
max_time=400


#SFM simulation engine introduced by Helbing et al. (1995) and coded by P. Wang, group-social-force, GitHub repository
# (2016). https://github.com/godisreal/group-social-force
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
def g(x):
    return np.max(x, 0)
def vectorAngleCos(x, y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same dimension')
        return
    angleCos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    angle = np.arccos(angleCos)
    return angle
def distanceP2W(point, wall):
    p0 = np.array([wall[0], wall[1]])
    p1 = np.array([wall[2], wall[3]])
    d = p1 - p0
    ymp0 = point - p0
    ymp1 = point - p1
    t = np.dot(d, ymp0) / np.dot(d, d)
    if t <= 0.0:
        dist = np.sqrt(np.dot(ymp0, ymp0))
        cross = p0 + t * d
    elif t >= 1.0:
        # ymp1 = point-p1
        dist = np.sqrt(np.dot(ymp1, ymp1))
        cross = p0 + t * d
    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
    npw = normalize(cross - point)
    return dist, npw
class Pedestrian(object):
    def __init__(self, x=1, y=1):
        self.posX = random.uniform(80, 80)
        self.posY = random.uniform(80, 80)
        self.pos = np.array([self.posX, self.posY])
        self.actualVX = 10.0  # random.uniform(0,1.6)
        self.actualVY = 10.0  # random.uniform(0,1.6)
        self.actualV = np.array([self.actualVX, self.actualVY])
        self.dest = np.array([80.0, 100.0])
        self.direction = normalize(self.dest - self.pos)
        self.desiredSpeed = 1.5  # random.uniform(0.3,2.3) #1.8
        self.desiredV = self.desiredSpeed * self.direction
        self.acclTime = 0.5  # random.uniform(8,16) #10.0
        self.drivenAcc = (self.desiredV - self.actualV) / self.acclTime
        self.mass = 60  # random.uniform(40,90) #60.0
        self.radius = 0.35
        self.interactionRange = 1.2
        self.p = 0.0
        self.bodyFactor = 120000
        self.slideFricFactor = 240000
        self.A = 1.5
        self.B = 0.07  # random.uniform(0.8,1.6) #0.8 #0.08
        self.desiredV_old = np.array([0.0, 0.0])
        self.actualV_old = np.array([0.0, 0.0])
    def step(self):
        v0 = self.actualV
        r0 = self.pos
        self.direction = normalize(self.dest - self.pos)
        adapt = self.adaptVel()
        peopleInter = self.peopleInteraction()
        wallInter = self.wallInteraction()
        sumForce = adapt + peopleInter + wallInter
        accl = sumForce/self.mass
        self.actualV = v0 + accl
        self.pos = r0 + v0 + 0.5*accl
    def adaptVel(self):
        deltaV = self.desiredV - self.actualV
        if np.allclose(deltaV, np.zeros(2)):
            deltaV = np.zeros(2)
        return deltaV * self.mass / self.acclTime
    def selfRepulsion(self, Dfactor=1, Afactor=1, Bfactor=1):
        first = -self.direction * Afactor * self.A * np.exp((self.radius * Dfactor) / (self.B * Bfactor)) * (
                    self.radius * Dfactor)
        return first
    def changeAttr(self, x=1, y=1, Vx=1, Vy=1):
        self.posX = x
        self.posY = y
        self.pos = np.array([self.posX, self.posY])
        self.actualVX = Vx
        self.actualVY = Vy
        self.actualV = np.array([self.actualVX, self.actualVY])
    def showAttr(self):
        print('X and Y Position:', self.pos)
        print('The destination', self.dest)
    def peopleInteraction(self, other, Dfactor=1, Afactor=1, Bfactor=1):
        rij = self.radius + other.radius
        dij = np.linalg.norm(self.pos - other.pos)
        nij = (self.pos - other.pos) / dij
        tij = np.array([-nij[1], nij[0]])
        dvij = np.dot((self.actualV - other.actualV), tij)
        first = Afactor * self.A * np.exp((rij * Dfactor - dij) / (self.B * Bfactor)) * nij
        + self.bodyFactor * g(rij - dij) * nij + self.slideFricFactor * g(rij - dij) * dvij * tij
        return first
    def wallInteraction(self, wall):
        ri = self.radius
        diw, niw = distanceP2W(self.pos, wall)
        tiw = np.array([-niw[0], niw[1]])
        first = -self.A * np.exp((ri - diw) / self.B) * niw * 100
        + self.bodyFactor * g(ri - diw) * niw + self.slideFricFactor * g(ri - diw) * (
                self.actualV * tiw) * tiw
        return first
    def wallOnRoute(self, wall):
        self.pos
        self.actualV
        return True
    def peopleInterOpinion(self, other, Dfactor=1, Afactor=1, Bfactor=1):
        rij = self.radius + other.radius
        dij = np.linalg.norm(self.pos - other.pos)
        otherDirection = np.array([0.0, 0.0])
        otherSpeed = 0.0
        num = 0
        otherV = np.array([0.0, 0.0])
        if dij < self.interactionRange:
            otherDirection = normalize(other.actualV)
            otherSpeed = np.linalg.norm(other.actualV)
            num = 1
            otherV = other.actualV
        return otherDirection, otherSpeed, num, otherV




#NSGA-II search engine propose by N. Srinivas, K. Deb, Muiltiobjective optimization using nondominated sorting in genetic
# algorithms, Evolutionary computation, and coded by H.A. Khan, NSGA-II, GitHub repository (2017). https://github.com/haris989/NSGA-II
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = 999999999999
    return sorted_list

def fast_non_dominated_sort(values1, values2,values3):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]
    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] > values3[p]) or (values1[q] >= values1[p] and values2[q] > values2[p] and values3[q] > values3[p]) or (values1[q] > values1[p] and values2[q] >= values2[p] and values3[q] > values3[p]) or (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] >= values3[p]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] > values3[q]) or (values1[p] >= values1[q] and values2[p] > values2[q] and values3[p] > values3[q]) or (values1[p] > values1[q] and values2[p] >= values2[q] and values3[p] > values3[q]) or (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front

def crowding_distance(values1, values2,values3, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    sorted3 = sort_by_values(front, values3[:])
    distance[0] = 99999999999999
    distance[len(front) - 1] = 999999999999999
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values1[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values2[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values3[sorted3[k+1]] - values3[sorted3[k-1]])/(max(values3)-min(values3)+0.001)
    return distance

def crossover(a,b,cr):
    tmp1=[]
    tmp1 = b[0:cr] + a[cr:]
    return tmp1

def crossover2(a,b,cr):
    tmp2=[]
    tmp2 = a[0:cr] + b[cr:]
    return tmp2

def mutation():
    aa = [random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)]
    return aa

# [[0,0,0,0,0]for j in range(int(pop_size/proc_no))]    FOR SIMULTANEOUS STRATEGY
solution_1=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_2=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_3=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_4=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_5=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_6=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_7=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]
solution_8=[[random.randrange(min_x, max_x,Step)  for i in range(num_of_zones)] for j in range(int(pop_size/proc_no))]

for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_1[i][j] == solution_1[i][k] and j != k:
                solution_1[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_2[i][j] == solution_2[i][k] and j != k:
                solution_2[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_3[i][j] == solution_3[i][k] and j != k:
                solution_3[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_4[i][j] == solution_4[i][k] and j != k:
                solution_4[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_5[i][j] == solution_5[i][k] and j != k:
                solution_5[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_6[i][j] == solution_6[i][k] and j != k:
                solution_6[i][k] = random.randrange(min_x, max_x, Step)
for i in range(int(pop_size/proc_no)):
    for j in range(num_of_zones):
        for k in range(num_of_zones):
            if solution_7[i][j] == solution_7[i][k] and j != k:
                solution_7[i][k] = random.randrange(min_x, max_x, Step)

gen_no=0
#due to multi-processing, def fi to def f8 are almost in a same format but every one execute a different set of evacuation scenarios (departure time patterns)
def f1(solution_1):
    congestion_values=[]
    fire_risk_values=[]
    TET_values=[]
    timer_resetting=0
    for i in range(len(solution_1)):
        Departure_Time = solution_1[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05*hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width*2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03*hall_lengh*1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width*2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03*hall_lengh*1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width -5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05*hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width-5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f1')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting==0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)


            zone1_x_start , zone1_y_start,zone1_x_end , zone1_y_end = hall_lengh*2/3,hall_width * 2 / 5, hall_lengh , hall_width * 2 / 5
            zone2_x_start , zone2_y_start,zone2_x_end , zone2_y_end = hall_lengh * 1 / 3 , hall_width * 2 / 5,hall_lengh * 2 / 3 , hall_width * 2 / 5
            zone3_x_start , zone3_y_start,zone3_x_end , zone3_y_end  = hall_lengh * 1 / 3 , hall_width * 2 / 5,hall_lengh * 1 / 3 , hall_width * 3 / 5
            zone4_x_start , zone4_y_start,zone4_x_end , zone4_y_end = hall_lengh * 1 / 3 , hall_width * 3 / 5,hall_lengh * 2 / 3 , hall_width * 3 / 5
            zone5_x_start , zone5_y_start,zone5_x_end , zone5_y_end = hall_lengh * 2 / 3 , hall_width * 3 / 5, hall_lengh , hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                #input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3 , hall_width * 3 / 5, hall_lengh-inner_door_width , hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed), hall_width]]#spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx,WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0), [walls[22][1]*ZOOMFACTOR, walls[22][1]*ZOOMFACTOR, walls[22][0]*ZOOMFACTOR, hall_width*ZOOMFACTOR-3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    #calculate crowd danger
                    if hall_width*2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03*hall_lengh/3  < ai.pos[0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width*2 / 5 < aj.pos[1] < hall_width * 3 / 5 and 1.03*hall_lengh/3  < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width*2 / 5 < aj.pos[1] < hall_width * 3 / 5 and hall_lengh  < ai.pos[0] < 1.22 * hall_lengh and hall_width*2 / 5 < ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0]==0:
                    speed=1.8
                else:
                    speed = ai.actualV[0]

                #calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22*hall_lengh - walls[22][0]) / (fire_speed) # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <=0:
                    t_estimate_dis= (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk=t_fire_exit-t_estimate_dis#max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0]<=1.22 * hall_lengh :
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0]<=1.22 * hall_lengh :
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                #determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET =  timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0],scPos[1]-AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]), AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits


def f2(solution_2):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_2)):
        Departure_Time = solution_2[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f2')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits


def f3(solution_3):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_3)):
        Departure_Time = solution_3[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f3')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits


def f4(solution_4):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_4)):
        Departure_Time = solution_4[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f4')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits


def f5(solution_5):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_5)):
        Departure_Time = solution_5[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f1')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits




def f6(solution_6):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_6)):
        Departure_Time = solution_6[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f6')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits


def f7(solution_7):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_7)):
        Departure_Time = solution_7[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f7')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits



def f8(solution_8):
    congestion_values = []
    fire_risk_values = []
    TET_values = []
    timer_resetting = 0
    for i in range(len(solution_8)):
        Departure_Time = solution_8[i]
        Congestion_Risk = 0
        fire_risk = 0
        realTET = 0
        agents = []
        for n in range(AGENTSNUM):
            agent = Pedestrian()
            agents.append(agent)
        for i in range(int(AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3), random.uniform(5, hall_width * 2 / 5)])
        for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(hall_lengh / 6, hall_lengh / 3),
                 random.uniform(hall_width / 3, 2 * hall_width / 3)])
        for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
            agents[i].pos = np.array(
                [random.uniform(1.03 * hall_lengh * 1 / 3, hall_lengh * 2 / 3),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
            agents[i].pos = np.array(
                [random.uniform(1.05 * hall_lengh * 2 / 3, hall_lengh),
                 random.uniform(hall_width * 3 / 5, hall_width - 5)])
        running = True
        while running:
            pygame.init()
            pygame.display.set_caption('f8')
            screen = pygame.display.set_mode(SCREENSIZE)
            clock = pygame.time.Clock()
            a = 0
            tt = pygame.time.get_ticks() / 1000
            if timer_resetting == 0:
                timeslap = copy.copy(tt)
            timer_resetting = 1
            timer = np.abs(tt - timeslap)
            screen.fill(BACKGROUNDCOLOR)

            zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = hall_lengh * 2 / 3, hall_width * 2 / 5, hall_lengh, hall_width * 2 / 5
            zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 2 / 3, hall_width * 2 / 5
            zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = hall_lengh * 1 / 3, hall_width * 2 / 5, hall_lengh * 1 / 3, hall_width * 3 / 5
            zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = hall_lengh * 1 / 3, hall_width * 3 / 5, hall_lengh * 2 / 3, hall_width * 3 / 5
            zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh, hall_width * 3 / 5

            for i in range(int(AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[0]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[0]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(AGENTSNUM / num_of_zones), int(2 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[1]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[1]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(2 * AGENTSNUM / num_of_zones), int(3 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[2]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[2]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(3 * AGENTSNUM / num_of_zones), int(4 * AGENTSNUM / num_of_zones)):
                if timer < Departure_Time[3]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[3]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])
            for i in range(int(4 * AGENTSNUM / num_of_zones), AGENTSNUM):
                if timer < Departure_Time[4]:
                    agents[i].dest = agents[i].pos
                elif timer >= Departure_Time[4]:
                    agents[i].dest = np.array([1.4 * hall_lengh, hall_width / 2])

            for idai, ai in enumerate(agents):
                density_around_i = 0.00
                Queue_num = 0.00
                if timer >= Departure_Time[0]:
                    zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[1]:
                    zone2_x_start, zone2_y_start, zone2_x_end, zone2_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[2]:
                    zone3_x_start, zone3_y_start, zone3_x_end, zone3_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[3]:
                    zone4_x_start, zone4_y_start, zone4_x_end, zone4_y_end = 1, 1, 2, 1
                if timer >= Departure_Time[4]:
                    zone5_x_start, zone5_y_start, zone5_x_end, zone5_y_end = 1, 1, 2, 1

                # input the facility walls' coordination, door placement, and spreading fire front position
                walls = [[1, 1, hall_lengh, 1],
                         [1, 1, 1, hall_width],
                         [1, hall_width, hall_lengh, hall_width],
                         [zone1_x_start, zone1_y_start, zone1_x_end, zone1_y_end],
                         [hall_lengh * 2 / 3, hall_width * 3 / 5, hall_lengh - inner_door_width, hall_width * 3 / 5],
                         [1 + (timer * fire_speed), 1, 1 + (timer * fire_speed),
                          hall_width]]  # spreading fire front position

                for wall in walls:
                    startPos = np.array([wall[0], wall[1]])
                    endPos = np.array([wall[2], wall[3]])
                    startPx = startPos * ZOOMFACTOR
                    endPx = endPos * ZOOMFACTOR
                    pygame.draw.line(screen, LINECOLOR, startPx, endPx, WALLTICKNESS)
                    pygame.draw.rect(screen, (255, 165, 0),
                                     [walls[22][1] * ZOOMFACTOR, walls[22][1] * ZOOMFACTOR, walls[22][0] * ZOOMFACTOR,
                                      hall_width * ZOOMFACTOR - 3])

                ai.direction = normalize(ai.dest - ai.pos)
                ai.desiredV = ai.desiredSpeed * ai.direction
                peopleInter = 0.0
                wallInter = 0.0
                otherMovingDir = np.array([0.0, 0.0])
                otherMovingSpeed = 0.0
                otherMovingNum = 0
                for idaj, aj in enumerate(agents):
                    if idai == idaj:
                        continue
                    peopleInter += ai.peopleInteraction(aj, Dfactor, Afactor, Bfactor)
                    rij = ai.radius + aj.radius
                    dij = np.linalg.norm(ai.pos - aj.pos)
                    # dij_dest = np.linalg.norm(ai.dest - aj.dest)
                    nij = (ai.pos - aj.pos) / dij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((ai.actualV - aj.actualV), tij)
                    vij_desiredV = np.linalg.norm(ai.desiredV - aj.desiredV)

                    # calculate crowd danger
                    if hall_width * 2 / 5 < ai.pos[1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < ai.pos[
                        0] < 1.22 * hall_lengh and ai.pos[0] < aj.pos[0] and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and 1.03 * hall_lengh / 3 < aj.pos[0] < 1.22 * hall_lengh:
                        Queue_num += 1.00
                    if hall_lengh < aj.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < aj.pos[
                        1] < hall_width * 3 / 5 and hall_lengh < ai.pos[0] < 1.22 * hall_lengh and hall_width * 2 / 5 < \
                            ai.pos[1] < hall_width * 3 / 5:
                        density_around_i += 1.00
                Congestion_Risk = Congestion_Risk + density_around_i * 3.00 * (
                            1.00 - np.e ** (-0.093 * density_around_i))

                if ai.actualV[0] == 0:
                    speed = 1.8
                else:
                    speed = ai.actualV[0]

                # calculate fire risk
                t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (speed)
                t_estimate_queue = Queue_num / door_pass_rate
                t_fire_exit = (1.22 * hall_lengh - walls[22][0]) / (
                    fire_speed)  # WALLS[] REFFERS TO THE WALL SHOWS THE FIRE FRONT
                if t_estimate_dis <= 0:
                    t_estimate_dis = (1.22 * hall_lengh - ai.pos[0]) / (0.05)
                delta_t_freewalk = t_fire_exit - t_estimate_dis  # max(t_estimate_dis,t_estimate_queue)
                delta_t_congestion = t_fire_exit - t_estimate_queue
                if delta_t_freewalk >= 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + (1.00 + 10000 / delta_t_freewalk)
                elif delta_t_freewalk < 0.1 and ai.pos[0] <= 1.22 * hall_lengh:
                    fire_risk_1sec = fire_risk_1sec + 100

                if otherMovingNum > 0:
                    ai.direction = (1 - ai.p) * ai.direction + ai.p * otherMovingDir
                    ai.desiredSpeed = (1 - ai.p) * ai.desiredSpeed + ai.p * otherMovingSpeed / otherMovingNum
                    ai.desiredV = ai.desiredSpeed * ai.direction
                adapt = ai.adaptVel()
                for wall in walls:
                    wallInter += ai.wallInteraction(wall)
                sumForce = adapt + peopleInter + wallInter
                accl = sumForce / ai.mass
                ai.actualV = ai.actualV + accl * tou  # consider dt = 0.5
                ai.pos = ai.pos + ai.actualV * tou
                ai.actualV_old = ai.actualV
                ai.desiredV_old = ai.desiredV

                # determine when entire the facility is evacuated anf process is finished
                if (int(ai.pos[0]) >= 1.22 * hall_lengh):
                    a += 1
            if a >= (AGENTSNUM) or timer > max_time:
                realTET = timer
                timeslap = tt
                running = False

            for agent in agents:
                scPos = [0, 0]
                scPos[0] = int(agent.pos[0] * ZOOMFACTOR)
                scPos[1] = int(agent.pos[1] * ZOOMFACTOR)
                endPosV = [0, 0]
                endPosV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.actualV[0] * ZOOMFACTOR)
                endPosV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.actualV[1] * ZOOMFACTOR)
                endPosDV = [0, 0]
                endPosDV[0] = int(agent.pos[0] * ZOOMFACTOR + agent.desiredV[0] * ZOOMFACTOR)
                endPosDV[1] = int(agent.pos[1] * ZOOMFACTOR + agent.desiredV[1] * ZOOMFACTOR)
                pygame.draw.ellipse(screen, AGENTCOLOR, (scPos[0], scPos[1] - AGENTSIZE[1], AGENTSIZE[0], AGENTSIZE[1]),
                                    AGENTSICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR1, scPos, endPosV, FORCELINETICKNESS)
                pygame.draw.line(screen, FORCELINECOLOR2, scPos, endPosDV, FORCELINETICKNESS)
            pygame.display.flip()
            clock.tick()
            plt.show()
        congestion_values.append(Congestion_Risk)
        fire_risk_values.append(fire_risk)
        TET_values.append(realTET)
    congestion_values_3digits = [float('{:.3f}'.format(value)) for value in congestion_values]
    fire_risk_values_3digits = [float('{:.3f}'.format(value)) for value in fire_risk_values]
    TET_values_3digits = [float('{:.3f}'.format(value)) for value in TET_values]
    pygame.quit()
    return TET_values_3digits, fire_risk_values_3digits, congestion_values_3digits




min_value_in_one_obj_set,gen_set,max_index_set, solution_fuzzy_set, obj1_fuzzy_set, obj2_fuzzy_set, obj3_fuzzy_set, max_obj1_set, min_obj1_set, avg_obj1_set, max_obj2_set, min_obj2_set, avg_obj2_set, max_obj3_set, min_obj3_set, avg_obj3_set ,total_obj1_value,total_obj2_value,total_obj3_value,total_solution= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
if __name__ == '__main__':
    pool = multiprocessing.Pool(proc_no)
    while (gen_no < max_gen):
        print('*** GENERATION NUMBER ***', gen_no)
        raw_function1_values = []
        raw_function2_values = []
        raw_function3_values = []
        raw_function1_values2 = []
        raw_function2_values2 = []
        raw_function3_values2 = []
        function1_values = []
        function2_values = []
        function3_values = []
        function1_values2 = []
        function2_values2 = []
        function3_values2 = []
        raw_solution = []
        raw_solution2 = []
        solution = []
        solution_9 = []
        solution_10 = []
        solution_11 = []
        solution_12 = []
        solution_13 = []
        solution_14 = []
        solution_15 = []
        solution_16 = []

        returns_f1 = pool.apply_async(f1, ([solution_1]))
        returns_f2 = pool.apply_async(f2, ([solution_2]))
        returns_f3 = pool.apply_async(f3, ([solution_3]))
        returns_f4 = pool.apply_async(f4, ([solution_4]))
        returns_f5 = pool.apply_async(f5, ([solution_5]))
        returns_f6 = pool.apply_async(f6, ([solution_6]))
        returns_f7 = pool.apply_async(f7, ([solution_7]))
        returns_f8 = pool.apply_async(f8, ([solution_8]))

        f1_output_TET, f1_output_fire, f1_output_congestion = returns_f1.get()
        f2_output_TET, f2_output_fire, f2_output_congestion = returns_f2.get()
        f3_output_TET, f3_output_fire, f3_output_congestion = returns_f3.get()
        f4_output_TET, f4_output_fire, f4_output_congestion = returns_f4.get()
        f5_output_TET, f5_output_fire, f5_output_congestion = returns_f5.get()
        f6_output_TET, f6_output_fire, f6_output_congestion = returns_f6.get()
        f7_output_TET, f7_output_fire, f7_output_congestion = returns_f7.get()
        f8_output_TET, f8_output_fire, f8_output_congestion = returns_f8.get()

        raw_function1_values.append(f1_output_TET)
        raw_function1_values.append(f2_output_TET)
        raw_function1_values.append(f3_output_TET)
        raw_function1_values.append(f4_output_TET)
        raw_function1_values.append(f5_output_TET)
        raw_function1_values.append(f6_output_TET)
        raw_function1_values.append(f7_output_TET)
        raw_function1_values.append(f8_output_TET)
        raw_function2_values.append(f1_output_fire)
        raw_function2_values.append(f2_output_fire)
        raw_function2_values.append(f3_output_fire)
        raw_function2_values.append(f4_output_fire)
        raw_function2_values.append(f5_output_fire)
        raw_function2_values.append(f6_output_fire)
        raw_function2_values.append(f7_output_fire)
        raw_function2_values.append(f8_output_fire)
        raw_function3_values.append(f1_output_congestion)
        raw_function3_values.append(f2_output_congestion)
        raw_function3_values.append(f3_output_congestion)
        raw_function3_values.append(f4_output_congestion)
        raw_function3_values.append(f5_output_congestion)
        raw_function3_values.append(f6_output_congestion)
        raw_function3_values.append(f7_output_congestion)
        raw_function3_values.append(f8_output_congestion)

        raw_solution.append(solution_1)
        raw_solution.append(solution_2)
        raw_solution.append(solution_3)
        raw_solution.append(solution_4)
        raw_solution.append(solution_5)
        raw_solution.append(solution_6)
        raw_solution.append(solution_7)
        raw_solution.append(solution_8)

        for i in range(proc_no):
            for j in range(int(pop_size / proc_no)):
                function1_values.append(raw_function1_values[i][j])
                function2_values.append(raw_function2_values[i][j])
                function3_values.append(raw_function3_values[i][j])
        for i in range(proc_no):
            for j in range(int(pop_size / proc_no)):
                solution.append(raw_solution[i][j])

        #NSGA-II (apply on set of parents)
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:],
                                                                function3_values[:])
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(function1_values[:], function2_values[:], function3_values[:],
                                  non_dominated_sorted_solution[i][:]))
        solution2 = solution.copy()

        while (len(solution2) < (2 * pop_size)):
            a1 = random.randint(0, pop_size - 1)
            b1 = random.randint(0, pop_size - 1)
            if a1 == b1:
                b1 = random.randint(0, pop_size - 1)
            if a1 == b1:
                b1 = random.randint(0, pop_size - 1)
            r = random.random()
            cr = random.randint(0, 4)
            if r >= 0.2:
                solution2.append(crossover(solution[a1], solution[b1], cr))
                if (len(solution2) == (2 * pop_size)):
                    break
                solution2.append(crossover2(solution[a1], solution[b1], cr))
            if r < 0.2:
                solution2.append(mutation())

        function1_values2 = function1_values.copy()
        function2_values2 = function2_values.copy()
        function3_values2 = function3_values.copy()

        for i in range(pop_size, int(pop_size + pop_size / proc_no)):
            solution_9.append(solution2[i])
        print('solution_9', solution_9)
        for i in range(int(pop_size + pop_size / proc_no), int(pop_size + pop_size * 2 / proc_no)):
            solution_10.append(solution2[i])
        for i in range(int(pop_size + pop_size * 2 / proc_no), int(pop_size + pop_size * 3 / proc_no)):
            solution_11.append(solution2[i])
        for i in range(int(pop_size + pop_size * 3 / proc_no), int(pop_size + pop_size * 4 / proc_no)):
            solution_12.append(solution2[i])
        for i in range(int(pop_size + pop_size * 4 / proc_no), int(pop_size + pop_size * 5 / proc_no)):
            solution_13.append(solution2[i])
        for i in range(int(pop_size + pop_size * 5 / proc_no), int(pop_size + pop_size * 6 / proc_no)):
            solution_14.append(solution2[i])
        for i in range(int(pop_size + pop_size * 6 / proc_no), int(pop_size + pop_size * 7 / proc_no)):
            solution_15.append(solution2[i])
        for i in range(int(pop_size + pop_size * 7 / proc_no), int(pop_size + pop_size * 8 / proc_no)):
            solution_16.append(solution2[i])

        returns_f1 = pool.apply_async(f1, ([solution_9]))
        returns_f2 = pool.apply_async(f2, ([solution_10]))
        returns_f3 = pool.apply_async(f3, ([solution_11]))
        returns_f4 = pool.apply_async(f4, ([solution_12]))
        returns_f5 = pool.apply_async(f5, ([solution_13]))
        returns_f6 = pool.apply_async(f6, ([solution_14]))
        returns_f7 = pool.apply_async(f7, ([solution_15]))
        returns_f8 = pool.apply_async(f8, ([solution_16]))

        f1_output_TET, f1_output_fire, f1_output_congestion = returns_f1.get()
        f2_output_TET, f2_output_fire, f2_output_congestion = returns_f2.get()
        f3_output_TET, f3_output_fire, f3_output_congestion = returns_f3.get()
        f4_output_TET, f4_output_fire, f4_output_congestion = returns_f4.get()
        f5_output_TET, f5_output_fire, f5_output_congestion = returns_f5.get()
        f6_output_TET, f6_output_fire, f6_output_congestion = returns_f6.get()
        f7_output_TET, f7_output_fire, f7_output_congestion = returns_f7.get()
        f8_output_TET, f8_output_fire, f8_output_congestion = returns_f8.get()

        raw_function1_values2.append(f1_output_TET)
        raw_function1_values2.append(f2_output_TET)
        raw_function1_values2.append(f3_output_TET)
        raw_function1_values2.append(f4_output_TET)
        raw_function1_values2.append(f5_output_TET)
        raw_function1_values2.append(f6_output_TET)
        raw_function1_values2.append(f7_output_TET)
        raw_function1_values2.append(f8_output_TET)
        raw_function2_values2.append(f1_output_fire)
        raw_function2_values2.append(f2_output_fire)
        raw_function2_values2.append(f3_output_fire)
        raw_function2_values2.append(f4_output_fire)
        raw_function2_values2.append(f5_output_fire)
        raw_function2_values2.append(f6_output_fire)
        raw_function2_values2.append(f7_output_fire)
        raw_function2_values2.append(f8_output_fire)
        raw_function3_values2.append(f1_output_congestion)
        raw_function3_values2.append(f2_output_congestion)
        raw_function3_values2.append(f3_output_congestion)
        raw_function3_values2.append(f4_output_congestion)
        raw_function3_values2.append(f5_output_congestion)
        raw_function3_values2.append(f6_output_congestion)
        raw_function3_values2.append(f7_output_congestion)
        raw_function3_values2.append(f8_output_congestion)

        for i in range(proc_no):
            for j in range(int(pop_size / proc_no)):
                function1_values2.append(raw_function1_values2[i][j])
                function2_values2.append(raw_function2_values2[i][j])
                function3_values2.append(raw_function3_values2[i][j])

        # NSGA-II (apply on set of combination of parents and offsprings)
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:],
                                                                 function3_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], function3_values2[:],
                                  non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        solution = [solution2[i] for i in new_solution]

        solution_1 = [solution[i] for i in range(0, int(pop_size / proc_no))]
        solution_2 = [solution[i] for i in range(int(pop_size / proc_no), int(pop_size * 2 / proc_no))]
        solution_3 = [solution[i] for i in range(int(pop_size * 2 / proc_no), int(pop_size * 3 / proc_no))]
        solution_4 = [solution[i] for i in range(int(pop_size * 3 / proc_no), int(pop_size * 4 / proc_no))]
        solution_5 = [solution[i] for i in range(int(pop_size * 4 / proc_no), int(pop_size * 5 / proc_no))]
        solution_6 = [solution[i] for i in range(int(pop_size * 5 / proc_no), int(pop_size * 6 / proc_no))]
        solution_7 = [solution[i] for i in range(int(pop_size * 6 / proc_no), int(pop_size * 7 / proc_no))]
        solution_8 = [solution[i] for i in range(int(pop_size * 7 / proc_no), int(pop_size * 8 / proc_no))]


        #  AVERAGING THE SAME SOLUTIONS
        obj1_value = [function1_values2[i] for i in new_solution]
        obj2_value = [function2_values2[j] for j in new_solution]
        obj3_value = [function3_values2[k] for k in new_solution]
        total_obj1_value.append(obj1_value)
        total_obj2_value.append(obj2_value)
        total_obj3_value.append(obj3_value)
        total_solution.append(solution)
        same = [[i] for i in range(len(obj1_value))]
        obj1_same = [[] for i in range(len(obj1_value))]
        obj2_same = [[] for i in range(len(obj1_value))]
        obj3_same = [[] for i in range(len(obj1_value))]
        solution_front = []
        obj1_front = []
        obj2_front = []
        obj3_front = []
        a = [0 for i in range(len(obj1_value))]

        for i in range(len(solution)):
            for k in range(0, i):
                if solution[i] == solution[k]:
                    a[i] = 1
                    same[i].remove(same[i][0])
                    break
            if a[i] == 0:
                for j in range(len(solution)):
                    if solution[i] == solution[j] and (i) < (j):
                        same[i].append(j)

        for i in range(len(same)):
            if same[i] != []:
                for j in same[i]:
                    obj1_same[i].append(obj1_value[j])
                    obj2_same[i].append(obj2_value[j])
                    obj3_same[i].append(obj3_value[j])
                obj1_front.append(np.average(obj1_same[i]))
                obj2_front.append(np.average(obj2_same[i]))
                obj3_front.append(np.average(obj3_same[i]))
                solution_front.append(solution[i])



                # USE TOPSIS TO FIND BEST SOLUTION AMONG PARETO SOLUTIONS
                # Y.-J. Lai, T.-Y. Liu, C.-L. Hwang, Topsis for MODM, European journal of operational research,
                max_function1_values2, min_function1_values2 = max(obj1_front), min(obj1_front)
                max_function2_values2, min_function2_values2 = max(obj2_front), min(obj2_front)
                max_function3_values2, min_function3_values2 = max(obj3_front), min(obj3_front)
                fuzzy = [[0] * int(len(obj1_front)) for i in range(3)]
                for value in range(len(obj1_front)):
                    fuzzy[0][value] = np.abs(obj1_front[value] - max_function1_values2) / (
                            max_function1_values2 - min_function1_values2)
                for value in range(len(obj2_front)):
                    fuzzy[1][value] = np.abs(obj2_front[value] - max_function2_values2) / (
                            max_function2_values2 - min_function2_values2)
                for value in range(len(obj3_front)):
                    fuzzy[2][value] = np.abs(obj3_front[value] - max_function3_values2) / (
                            max_function3_values2 - min_function3_values2)
                print('fuzzy', fuzzy)

                values_in_one_obj = [[] for i in range(len(obj1_front))]
                min_value_in_one_obj = []

                for value in range(len(obj1_front)):
                    for obj in range(3):
                        values_in_one_obj[value].append(fuzzy[obj][value])
                for value in range(len(obj1_front)):
                    min_value_in_one_obj.append(min(values_in_one_obj[value]))

                max_value_from_min_value = max(min_value_in_one_obj)
                max_index = min_value_in_one_obj.index(max_value_from_min_value)

                max_index_set.append(max_index)
                solution_fuzzy_set.append(solution_front[max_index])
                obj1_fuzzy_set.append(obj1_front[max_index])
                obj2_fuzzy_set.append(obj2_front[max_index])
                obj3_fuzzy_set.append(obj3_front[max_index])
                max_obj1_set.append(max(obj1_front))
                min_obj1_set.append(min(obj1_front))
                avg_obj1_set.append(np.average(obj1_front))
                max_obj2_set.append(max(obj1_front))
                min_obj2_set.append(min(obj2_front))
                avg_obj2_set.append(np.average(obj3_front))
                max_obj3_set.append(max(obj3_front))
                min_obj3_set.append(min(obj3_front))
                avg_obj3_set.append(np.average(obj3_front))
                min_value_in_one_obj_set.append(min_value_in_one_obj)


        # SAVE RESULT TO FILE
        with open('evoluted solution_GEN NO ' + str(gen_no) + ' .txt', 'w+') as f:
            print(solution, file=f)
        with open('evoluted TET values_GEN NO' + str(gen_no) + ' .txt', 'w+') as f:
            for i in new_solution:
                print(function1_values2[i], file=f)
        with open('evoluted fire values_GEN NO' + str(gen_no) + ' .txt', 'w+') as f:
            for i in new_solution:
                print(function2_values2[i], file=f)
        with open('evoluted congestion values_GEN NO' + str(gen_no) + ' .txt', 'w+') as f:
            for i in new_solution:
                print(function3_values2[i], file=f)
        with open('solution fuzzy' + ' .txt', 'w+') as f:
            print(solution2[max_index], file=f)
        with open('TET fuzzy'+ ' .txt', 'w+') as f:
            print(obj1_fuzzy_set, file=f)
        with open('fire fuzzy'  + ' .txt', 'w+') as f:
            print(obj2_fuzzy_set, file=f)
        with open('con fuzzy'  + ' .txt', 'w+') as f:
            print(obj3_fuzzy_set, file=f)
        with open('index fuzzy'  + ' .txt', 'w+') as f:
            print(max_index_set, file=f)
        with open('max_obj1_set'  + ' .txt', 'w+') as f:
            print(max_obj1_set, file=f)
        with open('min_obj1_set'  + ' .txt', 'w+') as f:
            print(min_obj1_set, file=f)
        with open('avg_obj1_set'  + ' .txt', 'w+') as f:
            print(avg_obj1_set, file=f)
        with open('max_obj2_set'  + ' .txt', 'w+') as f:
            print(max_obj2_set, file=f)
        with open('min_obj2_set'  + ' .txt', 'w+') as f:
            print(min_obj2_set, file=f)
        with open('avg_obj2_set'  + ' .txt', 'w+') as f:
            print(avg_obj2_set, file=f)
        with open('max_obj3_set'  + ' .txt', 'w+') as f:
            print(max_obj3_set, file=f)
        with open('min_obj3_set'  + ' .txt', 'w+') as f:
            print(min_obj3_set, file=f)
        with open('avg_obj3_set'  + ' .txt', 'w+') as f:
            print(avg_obj3_set, file=f)
        with open('min_value_in_one_obj_set' + ' .txt', 'w+') as f:
            print(min_value_in_one_obj_set, file=f)
        with open('solution_front' + str(gen_no) + ' .txt', 'w+') as f:
            print(solution_front, file=f)
        with open('obj1_front' + str(gen_no) + ' .txt', 'w+') as f:
            print(obj1_front, file=f)
        with open('obj2_front'+ str(gen_no) + ' .txt', 'w+') as f:
            print(obj2_front, file=f)
        with open('obj3_front'+ str(gen_no)  + ' .txt', 'w+') as f:
            print(obj3_front, file=f)
        with open('total_obj1_value'  + ' .txt', 'w+') as f:
            print(total_obj1_value, file=f)
        with open('total_obj2_value' + ' .txt', 'w+') as f:
            print(total_obj2_value, file=f)
        with open('total_obj3_value'  + ' .txt', 'w+') as f:
            print(total_obj3_value, file=f)


        gen_no = gen_no + 1


