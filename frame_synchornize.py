import ptgctl
import ptgctl.util
from ptgctl import holoframe
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class dicque(OrderedDict):
    def __init__(self, *a, maxlen=0, **kw):
        self._max = maxlen
        super().__init__(*a, **kw)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self._max > 0 and len(self) > self._max:
            self.popitem(False)

    def closest(self, tms):
        k = min(self, key=lambda t: abs(tms-t), default=None)
        return k

class FrameSyncApp:
    def __init__(self):
        self.api = ptgctl.API(username="test",
                              password="test")

    @ptgctl.util.async2sync
    async def run(self, prefix=None, track_all_objects=True):
        data = {}

        in_sids = ['main', 'depthlt']
        obj_sid = 'detic:image:misc:for3d' if track_all_objects else 'detic:image:for3d'
        reset_sids = ['depthltCal']
        out_sid = 'detic:sync'

        rgb_frames = dicque(maxlen=20)
        depth_frames = dicque(maxlen=20)
        server_time_to_sensor_time = dicque(maxlen=20)

        async with self.api.data_pull_connect(in_sids + [obj_sid] + reset_sids, ack=True) as ws_pull,\
                    self.api.data_push_connect(out_sid, batch=True) as ws_push:
            while True:
                for sid, t, buffer in await ws_pull.recv_data():
                    tms = int(t.split('-')[0])

                    if sid in reset_sids:
                        rgb_frames = dicque(maxlen=20)
                        depth_frames = dicque(maxlen=20)
                        server_time_to_sensor_time = dicque(maxlen=20)     
                        logger.info("sync cleared")                  
                    elif sid == 'depthlt':
                        d = holoframe.load(buffer, only_header=True)
                        depth_frames[d['time']] = buffer
                    elif sid == 'main':
                        d = holoframe.load(buffer, only_header=True)
                        rgb_frames[d['time']] = buffer
                        server_time_to_sensor_time[tms] = d['time']
                    elif sid == obj_sid:
                        if tms not in server_time_to_sensor_time:
                            logger.info("tms:{} not found".format(tms))
                            continue
                        if len(depth_frames) == 0:
                            continue
                        sensor_time = server_time_to_sensor_time[tms]
                        depth_frame = depth_frames[depth_frames.closest(sensor_time)]
                        await ws_push.send_data([rgb_frames[sensor_time] + depth_frame + buffer], [out_sid], [t])
                    
if __name__ == '__main__':
    import fire
    fire.Fire(FrameSyncApp)
