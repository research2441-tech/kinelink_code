"""BVH parsing and forward kinematics for CMU motion-capture clips.

Produces, at 100 Hz, world-frame positions and rotation matrices of the
segments that carry the four body nodes (shoulder, chest, wrist, ankle).
CMU BVH units are 1/0.45 inch; Y is vertical.
"""
import numpy as np

UNIT = 0.0254 / 0.45          # metres per BVH unit (CMU convention)


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    n = a.shape[0]
    R = np.zeros((n, 3, 3))
    if axis == 'X':
        R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    elif axis == 'Y':
        R[:, 1, 1] = 1; R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c
    else:
        R[:, 2, 2] = 1; R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c
    return R


def parse(path):
    tok = open(path).read().split()
    i = 0
    joints, stack = [], []
    while tok[i] != 'MOTION':
        t = tok[i]
        if t in ('ROOT', 'JOINT'):
            joints.append({'name': tok[i + 1], 'parent': stack[-1] if stack else -1,
                           'offset': None, 'channels': []})
            i += 2
        elif t == 'End':
            joints.append({'name': joints[stack[-1]]['name'] + '_end', 'parent': stack[-1],
                           'offset': None, 'channels': []})
            i += 2
        elif t == '{':
            stack.append(len(joints) - 1); i += 1
        elif t == '}':
            stack.pop(); i += 1
        elif t == 'OFFSET':
            joints[stack[-1]]['offset'] = np.array(list(map(float, tok[i + 1:i + 4]))); i += 4
        elif t == 'CHANNELS':
            n = int(tok[i + 1]); joints[stack[-1]]['channels'] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    nf = int(tok[i + 2]); dt = float(tok[i + 5])
    vals = np.array(list(map(float, tok[i + 6:])))
    nch = sum(len(j['channels']) for j in joints)
    motion = vals[:nf * nch].reshape(nf, nch)
    return joints, motion, dt


def forward_kinematics(joints, motion):
    nf = motion.shape[0]
    pos = np.zeros((len(joints), nf, 3))
    rot = np.zeros((len(joints), nf, 3, 3))
    c = 0
    for k, j in enumerate(joints):
        R = np.tile(np.eye(3), (nf, 1, 1))
        p = np.zeros((nf, 3))
        for ch in j['channels']:
            v = motion[:, c]; c += 1
            if ch.endswith('position'):
                p['XYZ'.index(ch[0])] if False else None
                p[:, 'XYZ'.index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], v)
        off = j['offset'] * UNIT
        if j['parent'] < 0:
            pos[k] = p * UNIT + off
            rot[k] = R
        else:
            P = j['parent']
            pos[k] = pos[P] + np.einsum('nij,j->ni', rot[P], off)
            rot[k] = rot[P] @ R
    return pos, rot


def resample(x, t_old, t_new):
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, i]) for i in range(flat.shape[1])], 1)
    return out.reshape((len(t_new),) + x.shape[1:])


def orthonormalise(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def load_clip(path, fs=100.0):
    """Return dict of segment poses (Y-up converted to Z-up, metres) at fs Hz."""
    joints, motion, dt = parse(path)
    pos, rot = forward_kinematics(joints, motion)
    names = [j['name'] for j in joints]
    # Y-up (BVH) -> Z-up (room): x_room = x, y_room = -z_bvh, z_room = y_bvh
    C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    pos = np.einsum('ij,knj->kni', C, pos)
    rot = np.einsum('ij,knjl->knil', C, rot)
    t_old = np.arange(motion.shape[0]) * dt
    t_new = np.arange(0, t_old[-1], 1.0 / fs)
    need = ['Hips', 'Spine1', 'Neck', 'Head', 'RightArm', 'RightForeArm', 'RightHand',
            'RightLeg', 'RightFoot', 'LeftUpLeg', 'RightUpLeg', 'LeftArm']
    out = {'t': t_new}
    for n in need:
        k = names.index(n)
        out[n + '_p'] = resample(pos[k], t_old, t_new)
        out[n + '_R'] = orthonormalise(resample(rot[k], t_old, t_new))
    # floor the clip: lowest foot height over clip -> 0.05 m (ankle joint)
    zmin = np.percentile(out['RightFoot_p'][:, 2], 2)
    for n in need:
        out[n + '_p'][:, 2] += 0.08 - zmin
    return out
