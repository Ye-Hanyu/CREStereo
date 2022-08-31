import os

import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2
import time
import glob

from nets import Model


def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model


def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = mge.tensor(imgL).astype("float32")
    imgR = mge.tensor(imgR).astype("float32")

    imgL_dw2 = F.nn.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.nn.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy()

    return pred_disp

# 相机参数读取
class cam_cfg:
    def __init__(self, config_path):
        self.config_path = config_path
        camera_config = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
        self.size = camera_config.getNode("Size").mat().astype(int)[0]
        self.left_matrix = camera_config.getNode("KL").mat()
        self.right_matrix = camera_config.getNode("KR").mat()
        self.left_distortion = camera_config.getNode("DL").mat()
        self.right_distortion = camera_config.getNode("DR").mat()
        self.R = camera_config.getNode("R").mat()
        self.T = camera_config.getNode("T").mat()

def onmouse_pick_points(event, x, y, flags, param):
    xyz = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(xyz[y, x])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
    parser.add_argument(
        "--model_path",
        default="models/crestereo_eth3d.mge",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument(
        "--left", default="img/test/left.png", help="The path of left image."
    )
    parser.add_argument(
        "--right", default="img/test/right.png", help="The path of right image."
    )
    parser.add_argument(
        "--size",
        default="1024x1536",
        help="The image size for inference. Te default setting is 1024x1536. \
                        To evaluate on ETH3D Benchmark, use 768x1024 instead.",
    )
    parser.add_argument(
        "--output", default="disparity.png", help="The path of output disparity."
    )
    args = parser.parse_args()

    assert os.path.exists(args.model_path), "The model path do not exist."
    assert os.path.exists(args.left), "The left image path do not exist."
    assert os.path.exists(args.right), "The right image path do not exist."

    model_func = load_model(args.model_path)
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    assert left.shape == right.shape, "The input images have inconsistent shapes."

    # 读取相机参数
    camera_cfg = cam_cfg('/mnt/sda1/Stereo/config/1280_720_cam0_4mm.yml')
    # 双目极线校正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_cfg.left_matrix,
                                                                      camera_cfg.left_distortion,
                                                                      camera_cfg.right_matrix,
                                                                      camera_cfg.right_distortion, camera_cfg.size,
                                                                      camera_cfg.R,
                                                                      camera_cfg.T,
                                                                      flags=cv2.CALIB_ZERO_DISPARITY,
                                                                      alpha=-1)

    # 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
    left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_cfg.left_matrix, camera_cfg.left_distortion, R1, P1,
                                                       camera_cfg.size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_cfg.right_matrix, camera_cfg.right_distortion, R2, P2,
                                                         camera_cfg.size, cv2.CV_16SC2)

    # map_rev = -np.ones(left_map1.shape, dtype=np.int16)
    # t0 = time.time()
    # for y, list in enumerate(left_map1):
    #     for x, point in enumerate(list):
    #         if point[1] < left_map1.shape[0] and point[0] < left_map1.shape[1]:
    #             map_rev[point[1], point[0]] = (x, y)
    #         # print('1')
    # t1 = time.time()
    # print('remap_rev', (t1 - t0) * 1000)

    img_list = glob.glob(r"/mnt/sda1/Stereo/data/cam0/1280-720-4mm/test/*")
    for i, path in enumerate(sorted(img_list)):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        w = img.shape[1]
        img_L = img[:, :int(w / 2)]
        img_R = img[:, int(w / 2):]
        img_L = cv2.remap(img_L, left_map1, left_map2, cv2.INTER_LINEAR)
        img_R = cv2.remap(img_R, right_map1, right_map2, cv2.INTER_LINEAR)

        in_h, in_w = img_L.shape[:2]
        eval_h = 720
        eval_w = 1280
        left_img = cv2.resize(img_L, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(img_R, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        t1 = time.time()
        pred = inference(left_img[300:420,600:840,:], right_img[300:420,600:840,:], model_func, n_iter=2)
        t2 = time.time()
        print((t2 - t1)*1000, 'ms')
        t = float(in_w) / float(eval_w)
        # disp = cv2.resize(pred, (280, 480), interpolation=cv2.INTER_LINEAR) * t
        disp = pred
        # np.save('./disp.npy', disp)
        T = np.eye(4, dtype=float)  # 用于将ROI区域映射回原图所在位置，计算3D坐标
        T[0, 3] = 600  # x
        T[1, 3] = 300  # y
        xyz_d = cv2.reprojectImageTo3D(disp, Q @ T, handleMissingValues=False)
        WIN_NAME = 'disp'
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, xyz_d)

        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imshow(WIN_NAME, disp_vis)
        cv2.waitKey(0)
        # parent_path = os.path.abspath(os.path.join(args.output, os.pardir))
        # if not os.path.exists(parent_path):
        #     os.makedirs(parent_path)
        # cv2.imwrite(args.output, disp_vis)
        # print("Done! Result path:", os.path.abspath(args.output))

        a=0

