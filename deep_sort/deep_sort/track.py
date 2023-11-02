# vim: expandtab:ts=4:sw=4

import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Occluded = 4

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    @classmethod
    def calculate_intersection(cls, track1, track2):
        # Implement intersection calculation logic here
        # For example, if your tracks are in (x, y, width, height) format:
        x1, y1, w1, h1 = track1.to_tlwh()
        x2, y2, w2, h2 = track2.to_tlwh()
        x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_intersection * y_intersection
        return intersection

    @classmethod
    def calculate_union(cls, track1, track2):
        # Implement union calculation logic here
        # For example:
        area1 = track1.to_tlwh()[2] * track1.to_tlwh()[3]
        area2 = track2.to_tlwh()[2] * track2.to_tlwh()[3]
        union = area1 + area2 - cls.calculate_intersection(track1, track2)
        return union

    @classmethod
    def _iou(cls, track1, track2):
        intersection = cls.calculate_intersection(track1, track2)
        union = cls.calculate_union(track1, track2)
        iou = intersection / union
        return iou
    
    #intersection of track
    @classmethod
    def _iot(cls, track1, track2):
        intersection = cls.calculate_intersection(track1, track2)
        area = track2.to_tlwh()[2] * track2.to_tlwh()[3]
        iot = intersection / area
        return iot

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, similarity_threshold=1.0):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # if self.state == TrackState.Occluded:
        #     # Calculate similarity between the untracked object's feature
        #     # and the saved feature of the object that occluded it
        #     similarity = self.calculate_similarity(self.occluded_by.features[-1], detection.feature)
            
        #     if similarity < similarity_threshold:
        #         self.state = TrackState.Confirmed
        #         self.occluded_by = None
        #         self.mean, self.covariance = kf.update(
        #             self.mean, self.covariance, detection.to_xyah())
        #         self.features.append(detection.feature)

        #         self.hits += 1
        #         self.time_since_update = 0
        #         if self.hits < self._n_init:
        #             self.state = TrackState.Tentative
        # else:
        #     self.mean, self.covariance = kf.update(
        #         self.mean, self.covariance, detection.to_xyah())
        #     self.features.append(detection.feature)

        #     self.hits += 1
        #     self.time_since_update = 0
        #     if self.state == TrackState.Tentative and self.hits >= self._n_init:
        #         self.state = TrackState.Confirmed

        if self.state == TrackState.Occluded:
            self.state = TrackState.Confirmed
            self.occluded_by = None
            if self.hits < self._n_init:
                self.state = TrackState.Tentative

        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    
    def calculate_similarity(self, feature1, feature2):
        """Calculate similarity between two feature vectors.

        Parameters
        ----------
        feature1 : ndarray
            Feature vector 1.
        feature2 : ndarray
            Feature vector 2.

        Returns
        -------
        float
            Similarity score.

        """
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

    def mark_missed(self, kf, matches, tracks, detections, iou_threshold=0.5, iot_threshold=0.7):
        """Mark this track as missed (no association at the current time step).
        """

        max_iot = 0.0
        
        for track_idx, detection_idx in matches:
            iou = self._iou(self, tracks[track_idx])  # calculate intersection of union
            iot = self._iot(self, tracks[track_idx])  # calculate intersection of track[track_idx]
            if iou >= iou_threshold and iot > max_iot:
                max_iot = iot
                # Untracked object is contained by another object
                # Save the CNN feature of the other object for later comparison
                self.state = TrackState.Occluded
                self.occluded_by = tracks[track_idx]

                self.mean, self.covariance = kf.update(
                    self.mean, self.covariance, detections[detection_idx].to_xyah())
                self.features.append(detections[detection_idx].feature)

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            if self.state != TrackState.Occluded:
                self.state = TrackState.Deleted
            elif self.state == TrackState.Occluded and iot_threshold > max_iot:
                self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def is_occluded(self):
        """Returns True if this track is occluded"""
        return self.state == TrackState.Occluded
