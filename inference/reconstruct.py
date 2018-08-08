from __future__ import print_function
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from model import *
from utils import *
from ply import *
import sys
sys.path.append("./nndistance/")
from modules.nnd import NNDModule
distChamfer = NNDModule()
import global_variables
import trimesh

import pyigl as igl
from iglhelpers import p2e,e2p
val_loss = AverageValueMeter()

def regress(points):
    """
    search the latent space to global_variables. Optimize reconstruction using the Chamfer Distance
    :param points: input points to reconstruct
    :return pointsReconstructed: final reconstruction after optimisation
    """
    points = Variable(points.data, requires_grad=True)
    latent_code = global_variables.network.encoder(points)
    lrate = 0.001  # learning rate
    # define parameters to be optimised and optimiser
    input_param = nn.Parameter(latent_code.data, requires_grad=True)
    global_variables.optimizer = global_variables.optim.Adam([input_param], lr=lrate)
    loss = 10
    i = 0

    #learning loop
    while np.log(loss) > -9 and i < global_variables.opt.nepoch:
        global_variables.optimizer.zero_grad()
        pointsReconstructed = global_variables.network.decode(input_param)  # forward pass
        dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        global_variables.optimizer.step()
        loss = loss_net.data[0]
        i = i + 1
    with torch.no_grad():
        if global_variables.opt.HR:
            pointsReconstructed = global_variables.network.decode_full(input_param)  # forward pass
        else :
            pointsReconstructed = global_variables.network.decode(input_param)  # forward pass
    # print("loss reg : ", loss)
    return pointsReconstructed, loss

def rotation_matrix(theta, flip):
    rot_y_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, flip, 0], [- np.sin(theta), 0,  np.cos(theta)]]).astype(np.float32)
    rot_y_matrix = Variable(torch.from_numpy(rot_y_matrix).float()).cuda()
    return rot_y_matrix

def run(input, scalefactor):
    """
    :param input: input mesh to reconstruct optimally.
    :return: final reconstruction after optimisation
    """

    input, translation = center(input)
    if not global_variables.opt.HR:
        mesh_ref = global_variables.mesh_ref_LR
    else:
        mesh_ref = global_variables.mesh_ref

    ## Extract points and put them on GPU
    points = input.vertices
    if np.shape(points)[0] > 10000:
        random_sample = np.random.choice(np.shape(points)[0], size=10000)
    else:
        random_sample = np.arange(np.shape(points)[0])

    points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
    points = Variable(points)
    points = points.transpose(2, 1).contiguous()
    points = points.cuda()

    # Get a low resolution PC to find the best reconstruction after a rotation on the Y axis
    points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
    points_LR = Variable(points_LR)
    points_LR = points_LR.transpose(2, 1).contiguous()
    points_LR = points_LR.cuda()

    theta = 0
    flip_y = 1
    bestLoss = 10
    pointsReconstructed = global_variables.network(points_LR)
    dist1, dist2 = distChamfer(points_LR.transpose(2, 1).contiguous(), pointsReconstructed)
    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
    # print("loss : ",  loss_net.data[0], 0)
    # ---- Search best angle for best reconstruction on the Y axis---
    for flip_y in [-1, 1]:
        for theta in np.linspace(-np.pi, np.pi, global_variables.opt.num_angles):
            if global_variables.opt.num_angles == 1:
                theta = 0
            #  Rotate mesh by theta and renormalise
            rot_y_matrix = rotation_matrix(theta, flip_y)
            points2 = torch.matmul(rot_y_matrix, points_LR)
            mesh_vert = points2[0].transpose(1,0).detach().data
            bbox0 = torch.max(mesh_vert,dim=0)[0]
            bbox1 = torch.min(mesh_vert, dim=0)[0]
            norma = Variable((bbox0 + bbox1) / 2)

            norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
            points2[0] = points2[0] - norma2

            # reconstruct rotated mesh
            pointsReconstructed = global_variables.network(points2)
            dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)


            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            if loss_net < bestLoss:
                bestLoss = loss_net
                best_theta, best_flip_y = theta, flip_y
                # unrotate the mesh
                norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()
                pointsReconstructed[0] = pointsReconstructed[0] + norma3
                rot_y_matrix = rotation_matrix(-theta, flip_y)
                pointsReconstructed = torch.matmul(pointsReconstructed, rot_y_matrix.transpose(1,0))
                bestPoints = pointsReconstructed

    # print("best loss and angle : ", bestLoss.data[0], best_theta)
    val_loss.update(bestLoss.data[0])

    #START REGRESSION
    print("start regression...")

    # rotate with optimal angle
    rot_y_matrix = rotation_matrix(best_theta, best_flip_y)
    points2 = torch.matmul(rot_y_matrix, points)
    mesh_vert = points2[0].transpose(1,0).detach().data
    bbox0 = torch.max(mesh_vert,dim=0)[0]
    bbox1 = torch.min(mesh_vert, dim=0)[0]
    norma = Variable((bbox0 + bbox1) / 2)
    norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
    points2[0] = points2[0] - norma2
    pointsReconstructed1, final_loss = regress(points2)
    # unrotate with optimal angle
    norma3 = norma.unsqueeze(0).expand(pointsReconstructed1.size(1), 3).contiguous()
    rot_y_matrix = rotation_matrix(-best_theta, best_flip_y)
    pointsReconstructed1[0] = pointsReconstructed1[0] + norma3
    pointsReconstructed1 = torch.matmul(pointsReconstructed1, rot_y_matrix.transpose(1,0))

    # create optimal reconstruction
    print("... Done!")
    final_points = (pointsReconstructed1[0].data.cpu().numpy()  + translation)/scalefactor
    return final_points, final_loss

def save(mesh, mesh_color, path, red, green, blue):
    """
    Home-made function to save a ply file with colors. A bit hacky
    """
    to_write = mesh.vertices
    b = np.zeros((len(mesh.faces),4)) + 3
    b[:,1:] = np.array(mesh.faces)
    points2write = pd.DataFrame({
        'lst0Tite': to_write[:,0],
        'lst1Tite': to_write[:,1],
        'lst2Tite': to_write[:,2],
        'lst3Tite': red,
        'lst4Tite': green,
        'lst5Tite': blue,
        })
    write_ply(filename=path, points=points2write, as_text=True, text=False, faces = pd.DataFrame(b.astype(int)), color = True)

def reconstruct(input_p):
    """
    Recontruct a 3D shape by deforming a template
    :param input_p: input path
    :return: None (but save reconstruction)
    """
    input = trimesh.load(input_p, process=False)
    scalefactor = 1.0
    if global_variables.opt.scale:
        input, scalefactor = scale(input, global_variables.mesh_ref_LR) #scale input to have the same volume as mesh_ref_LR
    if global_variables.opt.clean:
        input = clean(input) #remove points that doesn't belong to any edges
    test_orientation(input)
    mesh, meshReg = run(input, scalefactor)

    if not global_variables.opt.HR:
        red = global_variables.red_LR
        green = global_variables.green_LR
        blue = global_variables.blue_LR
        mesh_ref = global_variables.mesh_ref_LR
    else:
        blue = global_variables.blue_HR
        red = global_variables.red_HR
        green = global_variables.green_HR
        mesh_ref = global_variables.mesh_ref

    #save(mesh, global_variables.mesh_ref_LR, input_p[:-4] + "InitialGuess.ply", global_variables.red_LR, global_variables.green_LR, global_variables.blue_LR )
    save(meshReg, mesh_ref, input_p[:-4] + "FinalReconstruction.ply",  red, green, blue)
    # Save optimal reconstruction

def pca_whiten(V):
    V -= np.mean(V,axis=0)
    PCA = sampled_pca(V)
    V = np.matmul(V ,np.linalg.inv(PCA))
    V = rescale_V(V)
    return V

def rescale_V(V):
    V -= np.min(V,axis=0)
    V /= np.max(V)
    return V

def sampled_pca(X):
    EVal, EVec = np.linalg.eig(np.matmul(X.transpose(),X))
    return EVec.transpose()

def reconstruct_npz(inname, outname):
    """
    Recontruct a 3D shape by deforming a template
    :param inname: input path
    :return: None (but save reconstruction)
    """
    if os.path.exists(outname):
        return
    with np.load(inname) as npl:
        V, F = npl['V'], npl['F']
        V = pca_whiten(V)
        max_axis = np.argmax((np.max(V,axis=0) - np.min(V,axis=0)))
        V = V[:, np.roll(np.arange(3), 1-max_axis)] # 1 means Y
        V *= 1.7
    assert (np.max(V,axis=0) - np.min(V,axis=0))[1] > 1.69
    while V.shape[0] < 10000:
        eV, eF = p2e(V), p2e(F)
        NV,NF = igl.eigen.MatrixXd(), igl.eigen.MatrixXi()
        igl.upsample(eV,eF, NV,NF)
        V,F = e2p(NV), e2p(NF)

    input = trimesh.Trimesh(vertices=V, faces = F, process=False)
    scalefactor = 1.0
    if global_variables.opt.scale:
        input, scalefactor = scale(input, global_variables.mesh_ref_LR) #scale input to have the same volume as mesh_ref_LR
    if global_variables.opt.clean:
        input = clean(input) #remove points that doesn't belong to any edges
    test_orientation(input)

    final_points, final_loss = run(input, scalefactor)

    npz_path = os.path.dirname(outname)
    if not os.path.exists(npz_path): os.makedirs(npz_path)
    np.savez(outname, V=final_points, l = final_loss)

