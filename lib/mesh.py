import numpy as np
import os
class Mesh(object):
    def __init__(self, filename):
        v = []
        f = []
        ft = []
        fn = []
        vt = []
        vn = []
        vc = []
        segm = dict()
        landm_raw_xyz = dict()
        currSegm = ''
        currLandm = ''
        with open(filename, 'r', buffering=2 ** 10) as fp:
            for line in fp:
                line = line.split()
                if len(line) > 0:
                    if line[0] == 'v':
                        v.append([float(x) for x in line[1:4]])
                        if len(line) == 7:
                            vc.append([float(x) for x in line[4:]])
                        if currLandm:
                            landm_raw_xyz[currLandm] = v[-1]
                            currLandm = ''
                    elif line[0] == 'vt':
                        vt.append([float(x) for x in line[1:]])
                    elif line[0] == 'vn':
                        vn.append([float(x) for x in line[1:]])
                    elif line[0] == 'f':
                        faces = [x.split('/') for x in line[1:]]
                        for iV in range(1, len(faces) - 1):  # trivially triangulate faces
                            f.append([int(faces[0][0]), int(faces[iV][0]), int(faces[iV + 1][0])])
                            if (len(faces[0]) > 1) and faces[0][1]:
                                ft.append([int(faces[0][1]), int(faces[iV][1]), int(faces[iV + 1][1])])
                            if (len(faces[0]) > 2) and faces[0][2]:
                                fn.append([int(faces[0][2]), int(faces[iV][2]), int(faces[iV + 1][2])])
                            if currSegm:
                                segm[currSegm].append(len(f) - 1)
                    elif line[0] == 'g':
                        currSegm = line[1]
                        if currSegm not in segm.keys():
                            segm[currSegm] = []
                    elif line[0] == '#landmark':
                        currLandm = line[1]
                    elif line[0] == 'mtllib':
                        self.materials_filepath = os.path.join(os.path.dirname(filename), line[1])
                        self.materials_file = open(self.materials_filepath, 'r').readlines()

        self.v = np.array(v)
        self.f = np.array(f) - 1
        if vt:
            self.vt = np.array(vt)
        if vn:
            self.vn = np.array(vn)
        if vc:
            self.vc = np.array(vc)
        if ft:
            self.ft = np.array(ft) - 1
        if fn:
            self.fn = np.array(fn) - 1
        self.segm = segm
        self.landm_raw_xyz = landm_raw_xyz
        self.recompute_landmark_indices()

        for line in self.materials_file:
            if line and line.split() and line.split()[0] == 'map_Ka':
                self.texture_filepath = os.path.abspath(os.path.join(os.path.dirname(filename), line.split()[1]))
    
    def recompute_landmark_indices(self, landmark_fname=None, safe_mode=True):
        filtered_landmarks = dict(
            filter(
                lambda e, : e[1] != [0.0, 0.0, 0.0],
                self.landm_raw_xyz.items()
            ) if (landmark_fname and safe_mode) else self.landm_raw_xyz.items())
        if len(filtered_landmarks) != len(self.landm_raw_xyz):
            print("WARNING: %d landmarks in file %s are positioned at (0.0, 0.0, 0.0) and were ignored" % (len(self.landm_raw_xyz) - len(filtered_landmarks), landmark_fname))
    
        self.landm = {}
        self.landm_regressors = {}
        if filtered_landmarks:
            landmark_names = list(filtered_landmarks.keys())
            closest_vertices, _ = self.closest_vertices(np.array(list(filtered_landmarks.values())))
            self.landm = dict(zip(landmark_names, closest_vertices))
            if len(self.f):
                face_indices, closest_points = self.closest_faces_and_points(np.array(list(filtered_landmarks.values())))
                vertex_indices, coefficients = self.barycentric_coordinates_for_points(closest_points, face_indices)
                self.landm_regressors = dict([(name, (vertex_indices[i], coefficients[i])) for i, name in enumerate(landmark_names)])
            else:
                self.landm_regressors = dict([(name, (np.array([closest_vertices[i]]), np.array([1.0]))) for i, name in enumerate(landmark_names)])
