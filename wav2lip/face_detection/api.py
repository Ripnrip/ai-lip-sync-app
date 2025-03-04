from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
from .detection import sfd
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device or 'mps' in device:
            torch.backends.cudnn.benchmark = True
            if 'mps' in device and verbose:
                print("Using Apple Silicon GPU (MPS) for face detection.")

        # Get the face detector
        #face_detector_module = __import__('from .detection. import' + face_detector,
        #                                  globals(), locals(), [face_detector], 0)
        #self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)
        try:
            self.face_detector = sfd.FaceDetector(device=device, verbose=verbose)
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            print("Falling back to CPU for face detection.")
            # If detection fails on GPU (MPS/CUDA), fall back to CPU
            self.device = 'cpu'
            self.face_detector = sfd.FaceDetector(device='cpu', verbose=verbose)

    def get_detections_for_batch(self, images):
        """
        Returns a list of bounding boxes for each image in the batch.
        If no face is detected, returns None for that image.
        """
        try:
            # Convert to RGB for face detection 
            images = images.copy()
            if images.shape[-1] == 3:
                images = images[..., ::-1]  # BGR to RGB
            
            # Get face detections
            detected_faces = self.face_detector.detect_from_batch(images)
            
            results = []
            for i, d in enumerate(detected_faces):
                if len(d) == 0:
                    # No face detected
                    results.append(None)
                    continue
                    
                # Use the first (highest confidence) face
                d = d[0]
                # Ensure values are valid
                d = np.clip(d, 0, None)
                
                # Extract coordinates
                try:
                    x1, y1, x2, y2 = map(int, d[:-1])
                    # Sanity check on coordinates
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                        print(f"Invalid face coordinates: {(x1, y1, x2, y2)}")
                        results.append(None)
                    else:
                        results.append((x1, y1, x2, y2))
                except Exception as e:
                    print(f"Error processing detection: {str(e)}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            print(f"Error in batch face detection: {str(e)}")
            # Return None for all images
            return [None] * len(images)