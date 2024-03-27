import argparse
import open3d as o3d
import open3d.core as o3c
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def draw_registration_result(source, target, transformation):
    source_temp = source
    target_temp = target

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp,
         target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])


def chamfer_dist(pcd_src, pcd_dst, norm='L2'):
    tree_dst = cKDTree(np.asarray(pcd_dst.points))
    dists_src2dst, inds = tree_dst.query(np.asarray(pcd_src.points), k=1)

    tree_src = cKDTree(np.asarray(pcd_src.points))
    dists_dst2src, inds = tree_src.query(np.asarray(pcd_dst.points), k=1)

    if norm == 'L2':
        return np.mean(dists_src2dst**2) + np.mean(dists_dst2src**2)
    elif norm == 'L1':
        return np.mean(np.abs(dists_src2dst)) + np.mean(np.abs(dists_dst2src))
    else:
        raise NotImplementedError()


def f1_score(pcd_src, pcd_dst, threshold=0.05):
    res_src2dst = o3d.pipelines.registration.evaluate_registration(
        pcd_src, pcd_dst, threshold)
    precision = res_src2dst.fitness

    res_dst2src = o3d.pipelines.registration.evaluate_registration(
        pcd_dst, pcd_src, threshold)
    recall = res_dst2src.fitness

    F_score = 2 * (precision * recall) / (precision + recall)

    return F_score, precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    DST = "./MESHES_RESULT"
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    N = args.src.split('/')[-1][:-4]
    pcd_src = o3d.io.read_point_cloud(args.src)
    pcd_src.estimate_normals()
    pcd_src.paint_uniform_color([1, 0, 0])
    pcd_dst = o3d.io.read_point_cloud(args.dst)
    pcd_dst.estimate_normals()
    pcd_src.paint_uniform_color([0, 0, 1])

    threshold = 0.05
    f_score, precision, recall = f1_score(pcd_src, pcd_dst, args.threshold)
    print('F-score: {}, precision: {}, recall: {}'.format(
        f_score, precision, recall))

    # Chamfer distance
    chamfer_l1 = chamfer_dist(pcd_src, pcd_dst, 'L1')
    print('Chamfer-L1: {}'.format(chamfer_l1))
    print("==================AFTER REGISTRATION=====================")
    trans_init = np.asarray([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.], 
                            [0.0, 0.0, 0.0, 1.0]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    #draw_registration_result(pcd_src, pcd_dst, reg_p2p.transformation)
    pcd_src.transform(reg_p2p.transformation)

    #o3d.visualization.draw([pcd_src, pcd_dst])
    o3d.io.write_point_cloud("{}/{}_aligned.ply".format(DST,N), pcd_src)
    # F-score
    f_score, precision, recall = f1_score(pcd_src, pcd_dst, args.threshold)
    print('F-score: {}, precision: {}, recall: {}'.format(
        f_score, precision, recall))

    # Chamfer distance
    chamfer_l1 = chamfer_dist(pcd_src, pcd_dst, 'L1')
    print('Chamfer-L1: {}'.format(chamfer_l1))