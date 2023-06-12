import numpy as np
from collections import deque, Counter, defaultdict
import heapq
import cv2

SCORE_THRESHOLD = 1.4
UNSEEN_PENALTY = 2
LABEL_WINDOW_SIZE = 9

BOUNDARY_OFFSET = 20


class PredictionEntry:
    def __init__(self, pos, label, confidence, bbox=None):
        self.pos = pos
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.hand = False

    def __repr__(self):
        return "pos: {}, label: {}, conf: {}".format(
            self.pos, self.label, self.confidence)


class MemoryEntry:
    def __init__(self, id, pos, label, confidence, timestamp, win_size, bbox = None):
        self.id = id
        self.pos = pos
        self.labels = deque([label])
        self.confidence = confidence
        self.last_seen = timestamp
        self.window_size = win_size
        self.bbox = bbox
        self.unseen_count = 0

    def update(self, pos, label, confidence, timestamp, bbox = None, from_hand = False):
        self.pos = self.pos * (1-confidence) + pos * confidence
        if not from_hand:
            self.labels.append(label)
            if len(self.labels) > self.window_size:
                self.labels.popleft()
        self.confidence += confidence
        self.last_seen = timestamp
        self.bbox = bbox
        self.unseen_count = 0

    def get_label(self):
        c = Counter(self.labels)
        return c.most_common(1)[0][0]

    def __repr__(self):
        return "id: {}, pos: {}, labels: {}, conf: {}, last_seen: {}".format(
            self.id, self.pos, self.labels, self.confidence, self.last_seen)

    def to_dict(self):
        return {'pos': self.pos.tolist(), 'id':self.id, 'label':self.get_label()}

class MemoryHand:
    def __init__(self):
        self.objects = {}
        self.id = 0
        self.score_threshold = SCORE_THRESHOLD
        self.unseen_penalty = UNSEEN_PENALTY
        self.window_size = LABEL_WINDOW_SIZE

        self.prev_obj_id = [None, None]
        self.last_obj_id = None

    def update(self, detections, confidence, timestamp, intrinsics, world2pv_transform, img_shape, has_hands, hand_obj_boxes, hand_obj_det_ids, hand_obj_poses):
        matching = {}
        matched_mem_key = set()
        # data association
        scores = []
        for idx, d in enumerate(detections):
            # if idx in hand_obj_det_ids:
            #     d.confidence = min(1, d.confidence + 0.5)
            for k, o in self.objects.items():
                score, hand_score = self.getScore(d, o)
                if score > self.score_threshold:
                    scores.append((-score, idx, k))

        # matching
        heapq.heapify(scores)
        while len(matching) < len(detections) and len(matched_mem_key) < len(self.objects) and scores:
            _, det_i, mem_key = heapq.heappop(scores)
            if det_i in matching or mem_key in matched_mem_key:
                continue
            matching[det_i] = mem_key
            matched_mem_key.add(mem_key)

        # hand matching
        hand_matched = [False, False]
        for i in range(2):
            has_hand, hand_obj_box, hand_obj_det_id, hand_obj_pos = has_hands[i], hand_obj_boxes[i], hand_obj_det_ids[i], hand_obj_poses[i]
            if has_hand:
                if self.prev_obj_id[i] is not None and self.prev_obj_id[i] not in matched_mem_key and checkInsideFOV(self.objects[self.prev_obj_id[i]].pos, intrinsics, world2pv_transform, img_shape):
                    if hand_obj_det_id is None and hand_obj_box is not None:
                        hand_matched[i] = True
                        self.objects[self.prev_obj_id[i]].update(hand_obj_pos, self.objects[self.prev_obj_id[i]].get_label(), 1, timestamp, bbox = hand_obj_box, from_hand = True)      
                        matched_mem_key.add(self.prev_obj_id[i])
                    elif hand_obj_det_id is not None and hand_obj_det_id not in matching:
                        hand_matched[i] = True
                        matching[hand_obj_det_id] = self.prev_obj_id[i]
                        matched_mem_key.add(self.prev_obj_id[i])

        for i in range(2):
            has_hand, hand_obj_box, hand_obj_det_id, hand_obj_pos = has_hands[i], hand_obj_boxes[i], hand_obj_det_ids[i], hand_obj_poses[i]
            if has_hand:
                if self.last_obj_id is not None and self.last_obj_id in self.objects and self.last_obj_id not in matched_mem_key:
                    if hand_obj_det_id is not None and hand_obj_det_id not in matching:
                        # if self.objects[self.last_obj_id].get_label() == detections[hand_obj_det_id].label:
                        if detections[hand_obj_det_id].label in self.objects[self.last_obj_id].labels:
                            matching[hand_obj_det_id] = self.last_obj_id
                            matched_mem_key.add(self.last_obj_id)
                            hand_matched[i] = True
                            self.prev_obj_id[i] = self.last_obj_id

        # update
        for det_i, mem_key in matching.items():
            d = detections[det_i]
            self.objects[mem_key].update(
                d.pos, d.label, confidence, timestamp, bbox = d.bbox)

        # unseen objects:
        to_remove = []
        for mem_k, mem_entry in self.objects.items():
            if mem_k not in matched_mem_key and checkInsideFOV(
                    mem_entry.pos, intrinsics, world2pv_transform, img_shape):
                mem_entry.confidence -= self.unseen_penalty
                mem_entry.unseen_count += 1
                if mem_entry.confidence < 0:
                    to_remove.append(mem_k)
                elif mem_entry.unseen_count > 5:
                    to_remove.append(mem_k)               
        for k in to_remove:
            del self.objects[k]

        # new objects:
        for det_i, d in enumerate(detections):
            if det_i not in matching and detections[det_i].confidence > 0.6:
                self.objects[self.id] = MemoryEntry(
                    self.id, d.pos, d.label, confidence, timestamp, self.window_size, bbox = d.bbox)
                matching[det_i] = self.id
                self.id += 1
        
        # update hand obj info
        for i in range(2):
            if not hand_matched[i]:
                hand_obj_det_id = hand_obj_det_ids[i]
                if hand_obj_det_id is not None and hand_obj_det_id in matching:
                    self.last_obj_id = self.prev_obj_id[i] = matching[hand_obj_det_id]
                else:
                    self.prev_obj_id[i] = None

        return [mem_k for mem_k, mem_entry in self.objects.items() if checkInsideFOV(mem_entry.pos, intrinsics, world2pv_transform, img_shape)]

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
        return pos_score + class_score, 0

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return min(2 * np.exp(5 * -np.linalg.norm(pred.pos - mem.pos)), 3)

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return pred.confidence * sum(pred.label == i for i in mem.labels) / len(mem.labels) * (0.6 + min(0.4, mem.confidence / 10))



class Memory:
    def __init__(self):
        self.objects = {}
        self.id = 0
        self.score_threshold = SCORE_THRESHOLD
        self.unseen_penalty = UNSEEN_PENALTY
        self.window_size = LABEL_WINDOW_SIZE

    def update(self, detections, confidence, timestamp, intrinsics, world2pv_transform, img_shape, hand=None):
        # E-step
        scores = []
        for idx, d in enumerate(detections):
            for k, o in self.objects.items():
                score, hand_score = self.getScore(d, o, hand)
                if score > self.score_threshold or (hand_score > 0.7 and d.confidence > 0.5):
                    scores.append((-score, idx, k))

        # M-step
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
                d.pos, d.label, confidence, timestamp, bbox = d.bbox)

        # unseen objects:
        to_remove = []
        for mem_k, mem_entry in self.objects.items():
            if mem_k not in matched_mem_key and checkInsideFOV(
                    mem_entry.pos, intrinsics, world2pv_transform, img_shape):
                mem_entry.confidence -= self.unseen_penalty
                mem_entry.unseen_count += 1
                if mem_entry.confidence < 0:
                    to_remove.append(mem_k)
                # elif mem_entry.unseen_count > 3:
                #     to_remove.append(mem_k)               
        for k in to_remove:
            del self.objects[k]

        # new objects:
        for det_i, d in enumerate(detections):
            if det_i not in matching and detections[det_i].confidence > 0.6:
                self.objects[self.id] = MemoryEntry(
                    self.id, d.pos, d.label, confidence, timestamp, self.window_size, bbox = d.bbox)
                self.id += 1

        return [mem_k for mem_k, mem_entry in self.objects.items() if checkInsideFOV(mem_entry.pos, intrinsics, world2pv_transform, img_shape)]

    def __str__(self):
        strs = ["num objects: {}".format(len(self.objects))]
        for obj in self.objects.values():
            strs.append(str(obj))
        return '\n'.join(strs)

    def getScore(self, pred: PredictionEntry, mem: MemoryEntry, hand):
        pos_score = self.getPositionScore(pred, mem)
        class_score = self.getLabelScore(pred, mem)
        return pos_score + class_score, pos_score * 0.1 + class_score if self.getNearHand(pred, hand) else 0

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return min(2 * np.exp(5 * -np.linalg.norm(pred.pos - mem.pos)), 3)

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return 1.6 * pred.confidence * sum(pred.label == i for i in mem.labels) / len(mem.labels) * (0.6 + min(0.4, mem.confidence / 10))

    def getNearHand(self, pred: PredictionEntry, hand):
        if hand == None:
            return False
        if any(i != 0 for i in hand[0]) and np.linalg.norm(pred.pos - hand[0]) < 0.1:
            return True
        if any(i != 0 for i in hand[1]) and np.linalg.norm(pred.pos - hand[1]) < 0.1:
            return True
        return False

class Memory2Stage:
    def __init__(self):
        self.objects = {}
        self.id = 0
        self.score_threshold = SCORE_THRESHOLD
        self.unseen_penalty = UNSEEN_PENALTY
        self.window_size = LABEL_WINDOW_SIZE

    def update(self, detections, confidence, timestamp, intrinsics, world2pv_transform, img_shape, hand=None):
        matching = {}
        matched_mem_key = set()

        # E-step
        scores = []
        for idx, d in enumerate(detections):
            if d.confidence <= 0.6:
                continue
            for k, o in self.objects.items():
                score, hand_score = self.getScore(d, o, hand)
                if score > self.score_threshold or (hand_score > 0.7 and d.confidence > 0.6):
                    scores.append((-score, idx, k))

        # M-step
        heapq.heapify(scores)
        while len(matching) < len(detections) and len(matched_mem_key) < len(self.objects) and scores:
            _, det_i, mem_key = heapq.heappop(scores)
            if det_i in matching or mem_key in matched_mem_key:
                continue
            matching[det_i] = mem_key
            matched_mem_key.add(mem_key)

        # E-step2
        scores = []
        for idx, d in enumerate(detections):
            if d.confidence > 0.6:
                continue
            for k, o in self.objects.items():
                score, hand_score = self.getScore(d, o, hand)
                if score > self.score_threshold:
                    scores.append((-score, idx, k))

        # M-step2
        heapq.heapify(scores)
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
                d.pos, d.label, confidence, timestamp, bbox = d.bbox)

        # unseen objects:
        to_remove = []
        for mem_k, mem_entry in self.objects.items():
            if mem_k not in matched_mem_key and checkInsideFOV(
                    mem_entry.pos, intrinsics, world2pv_transform, img_shape):
                mem_entry.confidence -= self.unseen_penalty
                mem_entry.unseen_count += 1
                if mem_entry.confidence < 0:
                    to_remove.append(mem_k)
                elif mem_entry.unseen_count > 3:
                    to_remove.append(mem_k)               
        for k in to_remove:
            del self.objects[k]

        # new objects:
        for det_i, d in enumerate(detections):
            if det_i not in matching and detections[det_i].confidence > 0.6:
                self.objects[self.id] = MemoryEntry(
                    self.id, d.pos, d.label, confidence, timestamp, self.window_size, bbox = d.bbox)
                self.id += 1

        return [mem_k for mem_k, mem_entry in self.objects.items() if checkInsideFOV(mem_entry.pos, intrinsics, world2pv_transform, img_shape)]

    def __str__(self):
        strs = ["num objects: {}".format(len(self.objects))]
        for obj in self.objects.values():
            strs.append(str(obj))
        return '\n'.join(strs)

    def getScore(self, pred: PredictionEntry, mem: MemoryEntry, hand):
        pos_score = self.getPositionScore(pred, mem)
        class_score = self.getLabelScore(pred, mem)
        return pos_score + class_score, pos_score * 0.1 + class_score if self.getNearHand(pred, hand) else 0

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return min(2 * np.exp(5 * -np.linalg.norm(pred.pos - mem.pos)), 3)

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return 1.1 * sum(pred.label == i for i in mem.labels) / len(mem.labels) * (0.6 + min(0.4, mem.confidence / 10))

    def getNearHand(self, pred: PredictionEntry, hand):
        if hand == None:
            return False
        if any(i != 0 for i in hand[0]) and np.linalg.norm(pred.pos - hand[0]) < 0.1:
            return True
        if any(i != 0 for i in hand[1]) and np.linalg.norm(pred.pos - hand[1]) < 0.1:
            return True
        return False

class BaselineMemory(Memory):
    def __init__(self):
        super().__init__()
        self.score_threshold = 0
        self.unseen_penalty = 1000
        self.window_size = 1

    def getScore(self, pred: PredictionEntry, mem: MemoryEntry, hand):
        return self.getPositionScore(pred, mem) * self.getLabelScore(pred, mem), False

    def getPositionScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return np.exp(-np.linalg.norm(pred.pos - mem.pos))

    def getLabelScore(self, pred: PredictionEntry, mem: MemoryEntry):
        return pred.label == mem.labels[0]           

rvec = np.zeros(3)
tvec = np.zeros(3)


def checkInsideFOV(pos, intrinsics, world2pv_transform, img_shape):
    p = world2pv_transform @ np.hstack((pos, [1]))
    if p[2] > 0:
        return False
    xy, _ = cv2.projectPoints(
        p[:3], rvec, tvec, intrinsics, None)
    xy = np.squeeze(xy)
    height, width = img_shape
    xy[0] = width - xy[0]
    return BOUNDARY_OFFSET <= xy[0] < width-BOUNDARY_OFFSET and BOUNDARY_OFFSET <= xy[1] < height-BOUNDARY_OFFSET

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
