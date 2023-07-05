import numpy as np
import utils
import impl
import ptgctl
import ptgctl.util
from ptgctl import holoframe
from collections import OrderedDict
from hand_detector import HandDetector
import json
import orjson
import struct

def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0:
            img_obj_id.append(-1)
            continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
    return img_obj_id

class Memory3DApp:
    def __init__(self):
        self.api = ptgctl.API(username="test",
                              password="test",
                              url="http://172.24.113.199:7890")

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        data = {}

        in_sids = ['detic:sync']
        reset_sids = ['event:recipe:id', 'depthltCal']
        output_sid = 'detic:memory'

        async with self.api.data_pull_connect(in_sids + reset_sids, ack=True) as ws_pull, \
                self.api.data_push_connect(output_sid, batch=True) as ws_push:
            mem = impl.MemoryHand()
            hand_detector = HandDetector()
            data.update(holoframe.load_all(self.api.data('depthltCal')))
            while True:
                for sid, t, buffer in await ws_pull.recv_data():
                    if sid in reset_sids:
                        mem = impl.MemoryHand()
                        print("memory cleared")
                        if sid == 'depthltCal':
                            data['depthltCal'] = holoframe.load(buffer)
                            print("depth calibration updated")
                        continue

                    tms = int(t.split('-')[0])

                    format_str = "<BBQIIII"
                    header_size = struct.calcsize(format_str)
                    decoded = struct.unpack(format_str, buffer[:header_size])

                    jpg_end = header_size + decoded[-2] + decoded[-1]
                    rgb_frame = holoframe.load(buffer[:jpg_end])

                    decoded = struct.unpack(format_str, buffer[jpg_end:jpg_end+header_size])
                    depth_end = jpg_end + header_size + decoded[-1] + decoded[-4] * decoded[-3] * decoded[-2]
                    depth_frame = holoframe.load(buffer[jpg_end:depth_end])

                    detic_result = json.loads(buffer[depth_end:])


                    height, width = rgb_frame['image'].shape[:2]

                    depth_points = utils.get_points_in_cam_space(
                        depth_frame['image'], data['depthltCal']['lut'])
                    xyz, _ = utils.cam2world(
                        depth_points, data['depthltCal']['rig2cam'], depth_frame['rig2world'])
                    pos_image, mask = utils.project_on_pv(
                        xyz, rgb_frame['image'], rgb_frame['cam2world'],
                        [rgb_frame['focalX'], rgb_frame['focalY']], [rgb_frame['principalX'], rgb_frame['principalY']])

                    nms_idx = impl.nms(
                        detic_result["objects"], rgb_frame['image'].shape[:2])
                    detections = []
                    new_nms_idx = []
                    for i in nms_idx:
                        o = detic_result["objects"][i]
                        label, xyxyn = o["label"], o["xyxyn"]
                        if label in {'person'}:
                            continue

                        y1, y2, x1, x2 = int(xyxyn[1]*height), int(xyxyn[3]*height), int(
                            xyxyn[0]*width), int(xyxyn[2]*width)
                        pos_obj = pos_image[y1:y2, x1:x2, :]
                        mask_obj = mask[y1:y2, x1:x2]
                        pos_obj = pos_obj[mask_obj]
                        if pos_obj.shape[0] == 0:
                            continue
                        pos_obj = pos_obj.mean(axis=0)
                        new_nms_idx.append(i)
                        detections.append(
                            impl.PredictionEntry(pos_obj, label, o["confidence"], bbox = [x1,y1,x2,y2]))
                    nms_idx = new_nms_idx      

                    obj_dets, hand_dets = hand_detector.predict(rgb_frame['image'][:,:,::-1])

                    has_hand = [False, False]
                    hand_boxes, d_idxs, hand_obj_poses = [None, None], [None, None], [None, None]

                    if hand_dets is not None and obj_dets is not None:
                        img_obj_id = filter_object(obj_dets, hand_dets)
                        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
                            lr = 1 if hand_dets[i, -1] != 0 else 0
                            j = img_obj_id[i]                  
                            hand_box = hand_boxes[lr] = obj_dets[j,:4]
                            d_idx = None
                            has_hand[lr] = True
                            if detections:
                                d_boxes = np.array([d.bbox for d in detections])
                                xx1 = np.maximum(hand_box[0], d_boxes[:, 0])
                                yy1 = np.maximum(hand_box[1], d_boxes[:, 1])
                                xx2 = np.minimum(hand_box[2], d_boxes[:, 2])
                                yy2 = np.minimum(hand_box[3], d_boxes[:, 3])
                                w = np.maximum(0, xx2 - xx1)
                                h = np.maximum(0, yy2 - yy1)
                                intersection = w * h
                                union = (hand_box[2] - hand_box[0]) * (hand_box[3] - hand_box[1]) + (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1]) - intersection
                                iou = intersection / union
                                idx = np.argmax(iou)
                                if iou[idx] > 0.7:
                                    d_idx = idx
                                    detections[d_idx].hand = True
                                    d_idxs[lr] = d_idx

                            if d_idx is None:
                                x1, y1, x2, y2 = obj_dets[0,:4].astype(int)
                                pos_obj = pos_image[y1:y2, x1:x2, :]
                                mask_obj = mask[y1:y2, x1:x2]
                                pos_obj = pos_obj[mask_obj]
                                if pos_obj.shape[0] != 0:
                                    hand_obj_poses[lr] = pos_obj.mean(axis=0)      

                    intrinsic_matrix = np.array([[rgb_frame['focalX'], 0, width-rgb_frame['principalX']], [
                        0, rgb_frame['focalY'], rgb_frame['principalY']], [0, 0, 1]])
                    vis = mem.update(detections, 1, rgb_frame['time'],
                                intrinsic_matrix, np.linalg.inv(rgb_frame['cam2world']), rgb_frame['image'].shape[:2], has_hand, hand_boxes, d_idxs, hand_obj_poses)
                    await ws_push.send_data([orjson.dumps(mem.to_list())], [output_sid], [t])   

                    

if __name__ == '__main__':
    import fire
    fire.Fire(Memory3DApp)
