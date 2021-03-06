import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)


import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from kuka2 import Kuka
import random
import pybullet_data
maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KukaCamGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               cam=False):
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._width = 341
    self._height = 256
    self._isDiscrete=isDiscrete
    self.terminated = 0
    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    
    observation_high = np.array([np.finfo(np.float32).max] * observationDim)    
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    if cam:
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 4))
    else:
        largeValObservation = 100
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high)

    self.viewer = None

  def _reset(self):
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    self.table=p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    
    xpos = 0.5 +0.2*random.random()
    ypos = 0 +0.25*random.random()
    ang = 3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.1,orn[0],orn[1],orn[2],orn[3])
            
    p.setGravity(0,0,-10)
    self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getCameraObservation(self):
     #camEyePos = [0.03,0.236,0.54]
     #distance = 1.06
     #pitch=-56
     #yaw = 258
     #roll=0
     #upAxisIndex = 2
     #camInfo = p.getDebugVisualizerCamera()
     #print("width,height")
     #print(camInfo[0])
     #print(camInfo[1])
     #print("viewMatrix")
     #print(camInfo[2])
     #print("projectionMatrix")
     #print(camInfo[3])
     #viewMat = camInfo[2]
     #viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos,distance,yaw, pitch,roll,upAxisIndex)

     # fixed camera
     # view_matrix = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
     # #projMatrix = camInfo[3]#[0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
     # proj_matrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
     state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex+2)
     pos = state[0]
     orn = state[1]
     roll,pitch,yaw = p.getEulerFromQuaternion(orn)
     m=p.getMatrixFromQuaternion(orn)
     view_matrix=np.array([[*m[0:3],0],[*m[3:6],0],[*m[6:9],0],[*pos,1.0]])


     #base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

     view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
       cameraTargetPosition=pos,
       distance=-0.001,
       yaw=yaw,
       pitch=pitch,
       roll=roll,
       upAxisIndex=1)
     proj_matrix = self._p.computeProjectionMatrixFOV(
       fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
       nearVal=0.1, farVal=100.0)
     print("rpy {}".format((roll,pitch,yaw),view_matrix))

     img_arr = p.getCameraImage(width=self._width,height=self._height,viewMatrix=view_matrix,projectionMatrix=proj_matrix)
     rgb=img_arr[2]
     np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
     return np_img_arr

  def getExtendedObservation(self):

    self._observation = self._kuka.getObservation()
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    # print("gripperEul")
    # print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn, blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    # print("projectedBlockPos2D")
    # print(projectedBlockPos2D)
    # print("blockEulerInGripper")
    # print(blockEulerInGripper)

    # we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation
  
  def _step(self, action):
    if (self._isDiscrete):
      dv = 0.01
      dx = [0,-dv,dv,0,0,0,0][action]
      dy = [0,0,0,-dv,dv,0,0][action]
      da = [0,0,0,0,0,-0.1,0.1][action]
      f = 0.3
      realAction = [dx,dy,-0.002,da,f]
    else:
      dv = 0.01
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.1
      f = 0.3
      realAction = [dx,dy,-0.002,da,f]

    return self.step2( realAction)
     
  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      #self._observation = self.getExtendedObservation()
      self._envStepCounter += 1

    self._observation = self.getExtendedObservation()
    if self._renders:
        time.sleep(self._timeStep)
    #print("self._envStepCounter")
    #print(self._envStepCounter)
    
    done = self._termination()
    reward = self._reward()
    #print("len=%r" % len(self._observation))
    
    return np.array(self._observation), reward, done, {}

  def _render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])

    view_matrix = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
               -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0,
               0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
    proj_matrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
                  -0.02000020071864128, 0.0]
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]
      
    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter>maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005 
    closestPoints = p.getClosestPoints(self.table, self._kuka.kukaUid,maxDist)
     
    if (len(closestPoints)):#(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1
      
      print("closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range (100):
        graspAction = [0,0,0.0001,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0
    
      for i in range (1000):
        graspAction = [0,0,0.001,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2]>0.5):
          break

    
      self._observation = self.getExtendedObservation()
      return True
    return False
    
  def _reward(self):
    
    #rewards is height of target object
    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid,self._kuka.kukaUid,1000, -1, self._kuka.kukaEndEffectorIndex) 

    reward = -1000    
    numPt = len(closestPoints)
    #print(numPt)
    if (numPt>0):
      #print("reward:")
      reward = -closestPoints[0][8]*10
    if (blockPos[2] >0.2):
      #print("grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      reward = reward+1000

    #print("reward")
    #print(reward)
    return reward

