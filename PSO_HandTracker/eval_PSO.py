import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
# import pymesh
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from matplotlib.patches import Circle
from PSO import *
from tqdm.notebook import tqdm
#from utils import *
# from NYUdataSet import *
# import deodr
from deodr.pytorch import CameraPytorch,Scene3DPytorch
from deodr.pytorch import TriMeshPytorch as TriMesh
from pytorch3d.io import load_obj
from NYUdataSet2 import *

def denormalize(depth_map,max_depth=256,cof=125):
    mask=depth_map==1
    depth_map_V=depth_map*125.0#+com[2]
    depth_map_V=depth_map_V.float()-torch.min(depth_map_V).float()+10
    depth_map_V[mask]=max_depth
    return depth_map_V

def Generate_Random_hand(ncomps=6,center_idx=-1, use_pca=True,batch_size=1):
    mano_layer = ManoLayer(mano_root='mano/models',side="right",center_idx=center_idx, use_pca=use_pca, ncomps=ncomps)
    random_shape = torch.rand(batch_size, 10)*2
    random_pose = torch.rand(batch_size, ncomps + 3)
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)
    return hand_verts,hand_joints,mano_layer.th_faces,random_pose,random_shape


def Generate_Random_handV2(ncomps=10,center_idx=-1, use_pca=True,batch_size=1):
    mano_layer = ManoLayer(mano_root='mano/models',side="right",center_idx=center_idx, use_pca=use_pca, ncomps=ncomps)
    random_shape = torch.tensor(np.random.uniform(low=-0.03,high=0.03,size=(1,10))).float()
    random_pose = torch.tensor(np.random.uniform(low=-2,high=2,size=(1,ncomps+3))).float()
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)
    return hand_verts,hand_joints,mano_layer.th_faces,random_shape,random_pose


class GraphicsRenderer:
    def __init__(self,height=480,width=640,intrinsic=np.array([[1, 0, 0. ], [0, 1, 0 ], [0, 0, 1.]]),
        extrinsic=np.array([[ 1.,  0.,  0., -0.], [ 0., 1.,  0., -0.],[ 0.,  0., 1.,  0]])):
        # this class, given a mesh, it renders it
        self.intrinsic=intrinsic
        self.extrinsic=extrinsic
        self.width=width
        self.height=height
        self.camera=CameraPytorch(extrinsic=extrinsic,intrinsic=intrinsic,width=width,height=height,distortion=None)
        self.scene = Scene3DPytorch()

    def render(self,vertices,faces,max_depth=0,depth_scale = 1):
        # vertices is torch.tensor of size (n,3), faces is a torch.tensor of size (m,3)
        self.mesh = TriMesh(  faces.numpy().copy() )
        self.scene.set_mesh(self.mesh)
        self.scene.max_depth = max_depth
        self.scene.set_background(np.full((self.height, self.width, 1), max_depth, dtype=np.float))
        self.mesh.set_vertices(vertices.double())

        depth = self.scene.render_depth(self.camera, width=self.width, height=self.height, depth_scale=depth_scale)
        return depth

    
def Normalize(vertices,hand_joints,scale=0.6,translate=torch.tensor([0,0,1])):
       # vertices is torch.tensor of size (n,3), faces is a torch.tensor of size (m,3)
       # hand_joints is of shape (m,3)
               
        Mins=torch.min(vertices.double(),dim=0)[0];Mins[2]=Mins[2]-1
        r=(torch.squeeze(vertices.double())-Mins)
        verts=r*scale+translate
               
        joints=hand_joints.double()-Mins
        joints=joints*scale+translate
        
        return verts,joints
    

    
renderer=GraphicsRenderer(height=128,width=128)

def compute_center(img_inp,max_depth=0):
    # img is a tensor of size (h,w)
    img_inp=img_inp.squeeze()
    X,Y=torch.where(img_inp!=max_depth)
    X_mean=torch.mean(X.float())
    Y_mean=torch.mean(Y.float())
    return (X_mean,Y_mean)

def L1lossClamped(a,b,minn,maxx):
    # both a and b should be torch tensor of the same size
    diff=abs(a-b)
    loss=torch.clamp(diff,min=minn,max=maxx)
    return torch.mean(loss)


def get_NYU_compatible_joints(mesh_verts,joints,selected_joints=[20,18,16,14,12,10,8,6,4,3],selected_verts=[[231,7] ,[44,85], [37,190],67]):#[44,37,67]):
    # mesh verts and joints of of size (m,3) and (n,3) respectively
    final=joints[selected_joints]
    for element in selected_verts:
        if type(element)==int:
            temp=mesh_verts[element].reshape(1,3)
        else:
            temp=(mesh_verts[element[0]].reshape(1,3)+mesh_verts[element[1]].reshape(1,3))/2
            
        final=torch.cat([final,temp],dim=0)
    return final 


def Normalize2(vertices,hand_joints,scale=0.6,translate=torch.tensor([0,0,1])):
       # vertices is torch.tensor of size (n,3), faces is a torch.tensor of size (m,3)
       # hand_joints is of shape (m,3)
        offset=torch.tensor([64,64,40])
        #Mins=torch.min(vertices.double(),dim=0)[0];Mins[2]=Mins[2]-1
        r=(vertices.double()-vertices[67])*scale+offset
        verts=r+translate
               
        joints=(hand_joints.double()-vertices[67])*scale+offset
        joints=joints+translate
        
        return verts,joints

max_depth=0
# global_scale=0.6
global_scale=0.68

hand_verts,hand_joints,mano_faces,pp,sh=Generate_Random_hand(ncomps=10)
gt_verts,gt_joints=Normalize(hand_verts[0],hand_joints[0],translate=torch.tensor([20,20,10]),scale=global_scale)
IMG=renderer.render(gt_verts,mano_faces,max_depth=max_depth)
nyu_joints=get_NYU_compatible_joints(gt_verts,gt_joints)


ncomps=14
mano_layer = ManoLayer(mano_root='mano/models',side="right",center_idx=-1, use_pca=True, ncomps=ncomps)

MSE=torch.nn.L1Loss()


def compute_IOU(inp1,inp2,max_depth=-32):
    # both are tensor of size (1,h,w)
    rend1=inp1.clone();rend2=inp2.clone()
    mask1=rend1!=max_depth
    mask2=rend2!=max_depth
    return torch.sum(torch.logical_and(mask1,mask2)).double()/torch.sum(torch.logical_or(mask1,mask2))

def cost(x):
    #shape=torch.from_numpy(x[0,0:10].reshape(1,-1)).float()
    shape=sh
    pose=torch.from_numpy(x[0,0:(ncomps + 3)].reshape(1,-1)).float()
#     trans=torch.tensor([10,10,1])#torch.from_numpy(x[0,(ncomps + 3+10):(ncomps + 3+10+2)].reshape(1,-1)).float()
    trans=torch.from_numpy(x[0,(ncomps + 3):(ncomps + 3 +3)].reshape(1,-1)).float()
    #scale=x[0,-1]
#     TT=torch.from_numpy(x[0,(ncomps + 3+10+3):(ncomps + 3+3+3+10)].reshape(1,-1)).float()
    hand_verts, hand_joints = mano_layer(pose, shape)
    verts,joints=Normalize(hand_verts[0],hand_joints[0],translate=torch.tensor([trans[0,0],trans[0,1],trans[0,2]]),scale=global_scale)
    img=renderer.render(verts,mano_faces,max_depth=max_depth)
    
    recons_loss=MSE(IMG,img).numpy()#L1lossClamped(IMG,img,-1,12).numpy()
    shape_loss=torch.norm(shape).item()
    pose_loss=torch.norm(pose).item()
    iou_loss=1-compute_IOU(img,IMG,max_depth).item()
    loss= recons_loss+pose_loss+4*iou_loss#+shape_loss
    return loss

#initial=[5,5]               # initial starting location [x1,x2...]
#bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
dim=(ncomps+3 + 3)

center_x,center_y=compute_center(IMG)
nit=np.random.rand(1,dim);nit[0,-3]=center_y.numpy();nit[0,-2]=center_x.numpy();
S=PSO(num_dimensions=dim,lower_bound=-2,higher_bound=40,num_particles=256)

gt=[]
optimized=[]
results=[]
num_experiment=250
for i in range(num_experiment):
    hand_verts,hand_joints,mano_faces,pp,sh=Generate_Random_hand(ncomps=10)
    gt_verts,gt_joints=Normalize(hand_verts[0],hand_joints[0],translate=torch.from_numpy( np.random.uniform(low=0,high=5,size=3)*np.array([4,4,2])),scale=global_scale)
    IMG=renderer.render(gt_verts,mano_faces,max_depth=max_depth)
    nyu_joints=get_NYU_compatible_joints(gt_verts,gt_joints)

    S=PSO(num_dimensions=dim,lower_bound=-2,higher_bound=40,num_particles=256)
    S.optimize(cost,150,1,interval_rand=5,proportion_rand=0.4,dims=[-3,-2,-1,0,1,2],weights=[20,20,10,3,3,3])
    
    final_solution=S.pos_best_g
    shape=sh
    pose=torch.from_numpy(final_solution[0,:(ncomps + 3)].reshape(1,-1)).float()
    trans=torch.from_numpy(final_solution[0,(ncomps + 3):(ncomps + 3 +3)].reshape(1,-1)).float()
    hand_verts, hand_joints = mano_layer(pose, shape)
    verts,joints=Normalize(hand_verts[0],hand_joints[0],translate=torch.tensor([trans[0,0],trans[0,1],trans[0,2]]),scale=global_scale)
    img=renderer.render(verts,mano_faces)
    error=(nyu_joints-get_NYU_compatible_joints(verts,joints))
    gt.append(IMG.clone())
    optimized.append(img.clone())
    results.append(error)
    if i%10==0:
        print(i)




