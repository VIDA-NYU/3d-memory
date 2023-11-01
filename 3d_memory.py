import numpy as np
import utils
import impl
import ptgctl
import ptgctl.util
from ptgctl import holoframe
import json
import orjson
import struct

format_str = "<BBQIIII"
header_size = struct.calcsize(format_str)


def decode_buffer(buffer):
    decoded = struct.unpack(format_str, buffer[:header_size])

    jpg_end = header_size + decoded[-2] + decoded[-1]
    rgb_frame = holoframe.load(buffer[:jpg_end])

    decoded = struct.unpack(format_str, buffer[jpg_end:jpg_end+header_size])
    depth_end = jpg_end + header_size + \
        decoded[-1] + decoded[-4] * decoded[-3] * decoded[-2]
    depth_frame = holoframe.load(buffer[jpg_end:depth_end])

    detic_result = json.loads(buffer[depth_end:])

    return rgb_frame, depth_frame, detic_result


class Memory3DApp:
    def __init__(self):
        self.api = ptgctl.API(username="test",
                              password="test",
                              url="http://172.24.113.199:7890")

    def reset_memory(self):
        self.mem = impl.Memory()
        print("memory initialized")

    def filter_detection(self, detection):
        if self.label_filtering and detection['label'] not in self.key_labels:
            return True
        if detection['label'] in self.ignore_labels:
            return True
        x1, y1, x2, y2 = detection['xyxy']
        if (x2-x1) * (y2-y1) < 200:
            return True
        return False

    def preprocess_detic_result(self, detic_result, img_shape):
        height, width = img_shape
        for o in detic_result:
            xyxyn = o["xyxyn"]
            y1, y2, x1, x2 = int(
                xyxyn[1]*height), int(xyxyn[3]*height), int(xyxyn[0]*width), int(xyxyn[2]*width)
            o['xyxy'] = [x1, y1, x2, y2]
        return [o for o in detic_result if not self.filter_detection(o)]

    @ptgctl.util.async2sync
    async def run(self, prefix=None, label_filtering=True):
        data = {}

        in_sids = ['detic:sync']
        reset_sids = ['depthltCal', 'arui:reset']
        output_sid = 'detic:memory'

        self.ignore_labels = {'toothpicks',
                              'jar_lid', 'person', 'banana_slice'}

        self.label_filtering = label_filtering
        self.key_labels = {'plate', 'bowl', 'microwave_oven', 'tortilla', 'tortilla_package'}

        async with self.api.data_pull_connect(in_sids + reset_sids, ack=True) as ws_pull, \
                self.api.data_push_connect(output_sid, batch=True) as ws_push:
            data.update(holoframe.load_all(self.api.data('depthltCal')))
            self.reset_memory()
            while True:
                for sid, t, buffer in await ws_pull.recv_data():
                    if sid in reset_sids:
                        self.reset_memory()
                        if sid == 'depthltCal':
                            data['depthltCal'] = holoframe.load(buffer)
                            print("depth calibration updated")
                        continue

                    tms = int(t.split('-')[0])

                    rgb_frame, depth_frame, detic_result = decode_buffer(
                        buffer)
                    height, width = rgb_frame['image'].shape[:2]
                    intrinsic_matrix = np.array([[rgb_frame['focalX'], 0, width-rgb_frame['principalX']], [
                                                0, rgb_frame['focalY'], rgb_frame['principalY']], [0, 0, 1]])

                    if detic_result['objects'] is None:
                        res = self.mem.interpolate(intrinsic_matrix, np.linalg.inv(
                            rgb_frame['cam2world']), rgb_frame['image'].shape[:2])
                    else:
                        depth_points = utils.get_points_in_cam_space(
                            depth_frame['image'], data['depthltCal']['lut'])
                        xyz, _ = utils.cam2world(
                            depth_points, data['depthltCal']['rig2cam'], depth_frame['rig2world'])
                        pos_image, mask = utils.project_on_pv(
                            xyz, rgb_frame['image'], rgb_frame['cam2world'],
                            [rgb_frame['focalX'], rgb_frame['focalY']], [rgb_frame['principalX'], rgb_frame['principalY']])

                        detic_result["objects"] = self.preprocess_detic_result(
                            detic_result["objects"], rgb_frame['image'].shape[:2])
                        detic_result["objects"] = [detic_result["objects"][i]
                                                   for i in impl.nms(detic_result["objects"])]

                        detections = []
                        for o in detic_result["objects"]:
                            pos_3d = impl.convert_detection_to_3d_pos(
                                o, pos_image, mask)
                            if pos_3d is None:
                                continue
                            detections.append(impl.PredictionEntry(
                                pos_3d, o['label'], o["confidence"], o))

                        res = self.mem.update(detections, rgb_frame['time'], intrinsic_matrix, np.linalg.inv(
                            rgb_frame['cam2world']), rgb_frame['image'].shape[:2])

                    await ws_push.send_data([orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY)], [output_sid], [t])


if __name__ == '__main__':
    import fire
    fire.Fire(Memory3DApp)
