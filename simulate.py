"""Place a motion-capture clip in a hospital room and compute node channels and IMU signals."""
import numpy as np
import bvh, rooms, raytrace as rt

NODES = ('shoulder', 'chest', 'wrist', 'ankle')
G = 9.81


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def node_poses(c):
    """Return dict node -> (position R x 3, normal R x 3, sensor rotation R x 3 x 3)."""
    S, H, K, F = c['Spine1_R'], c['RightHand_R'], c['RightLeg_R'], c['RightFoot_p']
    out = {}
    p = c['RightArm_p'] + np.einsum('nij,j->ni', S, [0, 0.06, 0])
    out['shoulder'] = (p, np.einsum('nij,j->ni', S, _unit(np.array([-0.35, 1.0, 0.0]))), S)
    p = c['Spine1_p'] + np.einsum('nij,j->ni', S, [0, 0.05, 0.19])
    out['chest'] = (p, np.einsum('nij,j->ni', S, _unit(np.array([0.0, 1.0, 1.0]))), S)
    nrm = np.einsum('nij,j->ni', H, [0, 1.0, 0])
    out['wrist'] = (c['RightHand_p'] + 0.03 * nrm, nrm, H)
    sh = _unit(c['RightLeg_p'] - F)
    p = F + 0.15 * sh + np.einsum('nij,j->ni', K, [0, 0, 0.06])
    out['ankle'] = (p, np.einsum('nij,j->ni', K, _unit(np.array([0.0, 0.5, 1.0]))), K)
    return out


def imu(p, R, fs=100.0):
    """Ideal gyroscope (rad/s) and accelerometer specific force (m/s^2) in the sensor frame."""
    dR = np.gradient(R, 1 / fs, axis=0)
    W = np.einsum('nji,njk->nik', R, dR)
    gyro = np.stack([W[:, 2, 1] - W[:, 1, 2], W[:, 0, 2] - W[:, 2, 0], W[:, 1, 0] - W[:, 0, 1]], 1) / 2
    acc_w = np.gradient(np.gradient(p, 1 / fs, axis=0), 1 / fs, axis=0)
    acc_w[:, 2] += G
    acc = np.einsum('nji,nj->ni', R, acc_w)
    return gyro, acc


def staff_paths(room, n, T, fs, rng):
    pos = np.zeros((n, T, 2))
    x0, y0, x1, y1 = room['walk']
    for s in range(n):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        tgt = p.copy(); wait = 0
        v = rng.uniform(0.7, 1.2)
        for t in range(T):
            if wait > 0:
                wait -= 1
            else:
                d = tgt - p
                if np.linalg.norm(d) < 0.05:
                    for _ in range(50):
                        tgt = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                        if rooms.walkable(room, tgt[None])[0]:
                            break
                    wait = int(rng.uniform(0, 2.0) * fs)
                else:
                    p = p + d / np.linalg.norm(d) * min(v / fs, np.linalg.norm(d))
            pos[s, t] = p
    return pos


def place(c, room, rng, min_len=1.0, fs=100.0, tries=40):
    xy = c['Hips_p'][:, :2]
    best = None
    for _ in range(tries):
        th = rng.uniform(0, 2 * np.pi)
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = (xy - xy.mean(0)) @ Rz.T
        x0, y0, x1, y1 = room['walk']
        off = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        ok = rooms.walkable(room, q + off)
        # longest contiguous valid run
        runs, s = [], None
        for i, v in enumerate(np.append(ok, False)):
            if v and s is None:
                s = i
            if not v and s is not None:
                runs.append((s, i)); s = None
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            if best is None or b - a > best[1] - best[0]:
                best = (a, b, th, off)
        if best and (best[1] - best[0]) >= 0.9 * len(xy):
            break
    a, b, th, off = best
    if (b - a) < min_len * fs:
        return None
    Rz3 = np.eye(3)
    Rz3[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    shift = np.zeros(3)
    shift[:2] = off - (xy.mean(0) @ Rz3[:2, :2].T)
    out = {'t': c['t'][a:b] - c['t'][a]}
    for k, v in c.items():
        if k.endswith('_p'):
            out[k] = v[a:b] @ Rz3.T + shift
        elif k.endswith('_R'):
            out[k] = np.einsum('ij,njk->nik', Rz3, v[a:b])
    return out


def simulate(path, room, n_staff, seed, chunk=200, fs=100.0):
    rng = np.random.default_rng(seed)
    c = bvh.load_clip(path, fs)
    c = place(c, room, rng)
    if c is None:
        return None
    T = len(c['t'])
    pc = rooms.patches(room)
    pre = rt.precompute(room, pc)
    staff = staff_paths(room, n_staff, T, fs, rng)
    poses = node_poses(c)
    # occluders: own torso, own head, staff bodies
    ca = [c['Hips_p'], c['Neck_p']]
    cb = [c['Neck_p'], c['Head_p'] + np.array([0, 0, 0.12])]
    cr = [0.15, 0.11]
    for s in range(n_staff):
        z0 = np.zeros((T, 3)); z0[:, :2] = staff[s]
        z1 = z0.copy(); z1[:, 2] = 1.75
        ca.append(z0); cb.append(z1); cr.append(0.22)
    ca, cb, cr = np.stack(ca, 1), np.stack(cb, 1), np.array(cr)
    res = {'t': c['t'], 'hips': c['Hips_p'], 'staff': staff}
    for nd in NODES:
        p, nrm, Rs = poses[nd]
        gyro, acc = imu(p, Rs, fs)
        res[nd + '_gyro'], res[nd + '_acc'] = gyro, acc
        res[nd + '_pos'], res[nd + '_nrm'] = p, nrm
        for link, fn in (('dl', rt.downlink), ('ul', rt.uplink)):
            H0, tau, Hf, isi, aux = [], [], [], [], []
            for i in range(0, T, chunk):
                sl = slice(i, min(T, i + chunk))
                h, ax = fn(room, pre, p[sl], nrm[sl], ca[sl], cb[sl], cr)
                s = rt.summarise(h, **rt.LINK[link])
                H0.append(s['H0']); tau.append(s['tau']); Hf.append(s['Hf']); isi.append(s['isi']); aux.append(ax)
            res[f'{nd}_{link}_H0'] = np.concatenate(H0)
            res[f'{nd}_{link}_tau'] = np.concatenate(tau)
            res[f'{nd}_{link}_Hf'] = np.abs(np.concatenate(Hf)).astype(np.float32)
            res[f'{nd}_{link}_isi'] = np.concatenate(isi).astype(np.float32)
            res[f'{nd}_{link}_aux'] = np.concatenate(aux)
    return res
