import numpy as np
import utils
import impl
import ptgctl
import ptgctl.util
from ptgctl import holoframe
import json
import orjson
import struct

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
            mem = impl.Memory()
            data.update(holoframe.load_all(self.api.data('depthltCal')))
            while True:
                for sid, t, buffer in await ws_pull.recv_data():
                    if sid in reset_sids:
                        mem = impl.Memory()
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

                    intrinsic_matrix = np.array([[rgb_frame['focalX'], 0, width-rgb_frame['principalX']], [
                        0, rgb_frame['focalY'], rgb_frame['principalY']], [0, 0, 1]])

                    mem.update(detections, rgb_frame['time'],
                                intrinsic_matrix, np.linalg.inv(rgb_frame['cam2world']), rgb_frame['image'].shape[:2])
                    mem_list = mem.to_list()

                    for tracklet in mem_list:
                        if tracklet["status"] != "outside":
                            xy = utils.project_pos_to_pv(tracklet['pos'], rgb_frame['cam2world'], intrinsic_matrix, width)
                            tracklet['xyxyn'] = [(xy[0]-30) / width, (xy[1]-30) / height, (xy[0]+30) / width, (xy[1]+30) / height]
                    await ws_push.send_data([orjson.dumps(mem_list, option=orjson.OPT_SERIALIZE_NUMPY)], [output_sid], [t])   

                    

if __name__ == '__main__':
    import fire
    fire.Fire(Memory3DApp)
