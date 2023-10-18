import numpy as np
from collections import deque, Counter, defaultdict
import heapq
import cv2
import utils

SCORE_THRESHOLD = 1.4
MAX_UNSEEN_COUNT = 5
LABEL_WINDOW_SIZE = 15
NEW_TRACKLET_THRESHOLD = 0.6

ALPHA = 5
LAMBDA = 2

BOUNDARY_OFFSET = 20
UNSEEN_PENALTY = 1

class PredictionEntry:
    def __init__(self, pos, label, confidence, bbox):
        self.pos = pos
        self.label = label
        self.confidence = confidence
        self.hand = False
        self.bbox = bbox

    def __repr__(self):
        return "pos: {}, label: {}, conf: {}".format(
            self.pos, self.label, self.confidence)


class MemoryEntry:
    def __init__(self, id, pos, label, timestamp, win_size, bbox):
        self.id = id
        self.bbox = bbox
        self.pos = pos
        self.label_count = Counter([label])
        self.labels = deque([label])
        self.last_seen = timestamp
        self.window_size = win_size
        self.count = 0
        self.unseen_count = 0
        self.status = 'tracked'

    def update(self, pos, label, timestamp, bbox):
        self.pos = pos
        self.bbox = bbox

        self.labels.append(label)
        self.label_count[label] += 1
        if len(self.labels) > self.window_size:
            l = self.labels.popleft()
            self.label_count[l] -= 1
            if self.label_count[l] == 0:
                del self.label_count[l]
 
        self.count += 1
        self.last_seen = timestamp
        self.unseen_count = 0
        self.status = 'tracked'

    def get_label(self):
        return self.label_count.most_common(1)[0][0]

    def __repr__(self):
        return "id: {}, pos: {}, labels: {}, conf: {}, last_seen: {}".format(
            self.id, self.pos, self.labels, self.count, self.last_seen)

    def to_dict(self):
        return {'pos': self.pos.tolist(), 'id': self.id, 'label': self.get_label(), 'last_seen': self.last_seen, 'status':self.status}


class Memory:
    def __init__(self):
        self.objects = {}
        self.id = 0
        self.score_threshold = SCORE_THRESHOLD
        self.unseen_penalty = UNSEEN_PENALTY
        self.window_size = LABEL_WINDOW_SIZE
        self.new_tracklet_threshold = NEW_TRACKLET_THRESHOLD
        self.max_unseen_count = MAX_UNSEEN_COUNT
        self.alpha = ALPHA
        self.beta = LAMBDA

    def update(self, detections, timestamp, intrinsics, world2pv_transform, img_shape, **kwargs):
        for i in self.objects.values():
            i.status = 'outside'

        # calculate similarity
        scores = []
        for idx, d in enumerate(detections):
            for k, o in self.objects.items():
                score = self.getScore(d, o)
                if score > self.score_threshold:
                    scores.append((-score, idx, k))

        # data association
        heapq.heapify(scores)
        matching = {}
        matched_mem_key = set()
        while len(matching) < len(detections) and len(matched_mem_key) < len(self.objects) and scores:
            _, det_i, mem_key = heapq.heappop(scores)
            if det_i in matching or mem_key in matched_mem_key:
                continue
            matching[det_i] = mem_key
            matched_mem_key.add(mem_key)

        # update
        for det_i, mem_key in matching.items():
            d = detections[det_i]
            self.objects[mem_key].update(
                d.pos, d.label, timestamp, d.bbox)

        # unseen objects:
        to_remove = []
        for mem_k, mem_entry in self.objects.items():
            if mem_k not in matched_mem_key and checkInsideFOV(
                    mem_entry.pos, intrinsics, world2pv_transform, img_shape):
                mem_entry.status = 'extended'
                mem_entry.count -= self.unseen_penalty
                mem_entry.unseen_count += 1
                if mem_entry.count < 0 or mem_entry.unseen_count > self.max_unseen_count:
                    to_remove.append(mem_k)
        for k in to_remove:
            del self.objects[k]

        # new objects:
        for det_i, d in enumerate(detections):
            if det_i not in matching and detections[det_i].confidence > self.new_tracklet_threshold:
                self.objects[self.id] = MemoryEntry(
                    self.id, d.pos, d.label, timestamp, self.window_size, d.bbox)
                self.id += 1


    def to_list(self):
        return [obj.to_dict() for obj in self.objects.values()]

    def __str__(self):
        strs = ["num objects: {}".format(len(self.objects))]
        for obj in self.objects.values():
            strs.append(str(obj))
        return '\n'.join(strs)

    def getScore(self, pred: PredictionEntry, mem: MemoryEntry):
        pos_score = self.getPositionScore(pred, mem)
        class_score = self.getLabelScore(pred, mem)
        return self.beta * pos_score + class_score

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return np.exp(self.alpha * -np.linalg.norm(pred.pos - mem.pos))

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return pred.confidence * mem.label_count[pred.label] / self.window_size


class BaselineMemory(Memory):
    def __init__(self):
        super().__init__()
        self.score_threshold = 0
        self.window_size = 1

    def getScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return self.getPositionScore(pred, mem) * self.getLabelScore(pred, mem)

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return np.exp(-np.linalg.norm(pred.pos - mem.pos))

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return pred.label == mem.labels[0]


_rvec = np.zeros(3)
_tvec = np.zeros(3)


def checkInsideFOV(pos, intrinsics, world2pv_transform, img_shape, boundary_offset=BOUNDARY_OFFSET):
    p = world2pv_transform @ np.hstack((pos, [1]))
    if p[2] > 0:
        return False
    xy, _ = cv2.projectPoints(
        p[:3], _rvec, _tvec, intrinsics, None)
    xy = np.squeeze(xy)
    height, width = img_shape
    xy[0] = width - xy[0]

    return boundary_offset <= xy[0] < width-boundary_offset and boundary_offset <= xy[1] < height-boundary_offset


def align_depth_to_rgb(img, img_json, depth_img, depth_json, depth_calibration):
    depth_points = utils.get_points_in_cam_space(
        depth_img, depth_calibration['lut'])
    xyz, _ = utils.cam2world(
        depth_points, depth_calibration['rig2cam'], depth_json['rig2world'])
    pos_image, mask = utils.project_on_pv(
        xyz, img, img_json['cam2world'],
        [img_json['focalX'], img_json['focalY']], [img_json['principalX'], img_json['principalY']])
    return pos_image, mask


def nms(results, shape, threshold=0.4):
    if len(results) == 0:
        return []
    height, width = shape
    boxes = np.zeros((len(results), 4))
    class_to_ids = defaultdict(list)
    scores = np.zeros(len(results))
    for i, res in enumerate(results):
        if res["label"] in {'person'}:
            continue
        class_to_ids[res["label"]].append(i)
        scores[i] = res["confidence"]
        xyxyn = res["xyxyn"]
        y1, y2, x1, x2 = int(
            xyxyn[1]*height), int(xyxyn[3]*height), int(xyxyn[0]*width), int(xyxyn[2]*width)
        boxes[i, :] = [x1, y1, x2, y2]

    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    # We add 1, because the pixel at the start as well as at the end counts
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    res = []
    for label, ids in class_to_ids.items():
        # The indices of all boxes at start. We will redundant indices one by one.
        if len(ids) == 1:
            res.append(ids[0])
            continue

        indices = np.array(sorted(ids, key=lambda i: scores[i]))
        while indices.size > 0:
            index = indices[-1]
            res.append(index)
            box = boxes[index, :]

            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0], boxes[indices[:-1], 0])
            yy1 = np.maximum(box[1], boxes[indices[:-1], 1])
            xx2 = np.minimum(box[2], boxes[indices[:-1], 2])
            yy2 = np.minimum(box[3], boxes[indices[:-1], 3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # compute the ratio of overlap
            ratio = intersection / areas[indices[:-1]]
            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            indices = indices[np.where(ratio < threshold)]

    # return only the boxes at the remaining indices
    return res
