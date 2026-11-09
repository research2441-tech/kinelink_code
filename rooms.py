"""Hospital room geometries: ICU ward and family-type patient room (FTPR).

Dimensions and luminaire counts follow the two settings used in the
original VLC-MBSN study (ICU 11.5 x 6.5 x 3 m, 15 luminaires; FTPR
7 x 5 x 3 m, 8 luminaires). Furniture is represented by axis-aligned boxes
that bound the walkable floor area and (for tall partitions) block rays.
"""
import numpy as np


def _grid(L, W, nx, ny, x0=0.0, y0=0.0, z=2.98):
    xs = x0 + L * (np.arange(nx) + 0.5) / nx
    ys = y0 + W * (np.arange(ny) + 0.5) / ny
    return np.array([[x, y, z] for x in xs for y in ys])


def icu():
    L, W, H = 11.5, 6.5, 3.0
    lum = _grid(L, W, 5, 3)
    beds = [(cx - 0.5, 0.1, 0.0, cx + 0.5, 2.2, 0.85) for cx in (1.6, 4.4, 7.1, 9.9)]
    desk = [(9.5, 5.5, 0.0, 11.3, 6.4, 1.1)]
    return dict(name='ICU', L=L, W=W, H=H, lum=lum, boxes=beds + desk, tall=[],
                walk=(0.5, 2.7, 11.0, 6.1), keepout=desk,
                rho=dict(ceiling=0.80, wall=0.69, floor=0.30))


def ftpr():
    L, W, H = 7.0, 5.0, 3.0
    # bathroom occupies the corner x in [0,2], y in [3,5]; luminaires over the remaining area
    lum = np.array([[2.9, 1.25, 2.98], [4.7, 1.25, 2.98], [6.2, 1.25, 2.98], [1.0, 1.25, 2.98],
                    [2.9, 3.75, 2.98], [4.7, 3.75, 2.98], [6.2, 3.75, 2.98], [1.0, 2.6, 2.98]])
    bed = [(2.5, 0.1, 0.0, 3.5, 2.2, 0.85)]
    sofa = [(5.0, 3.9, 0.0, 6.8, 4.8, 0.8)]
    bath = [(0.0, 3.0, 0.0, 2.0, 5.0, 3.0)]
    return dict(name='FTPR', L=L, W=W, H=H, lum=lum, boxes=bed + sofa + bath, tall=bath,
                walk=(0.4, 0.4, 6.6, 4.6), keepout=bed + sofa + bath,
                rho=dict(ceiling=0.80, wall=0.69, floor=0.30))


def patches(room, d=0.4):
    """Discretise the six room surfaces into square patches (centre, normal, area, rho)."""
    L, W, H = room['L'], room['W'], room['H']
    out = []

    def add(c, n, a, r):
        out.append((c, n, a, r))

    nx, ny, nz = int(np.ceil(L / d)), int(np.ceil(W / d)), int(np.ceil(H / d))
    dx, dy, dz = L / nx, W / ny, H / nz
    for i in range(nx):
        for j in range(ny):
            x, y = (i + 0.5) * dx, (j + 0.5) * dy
            inside_tall = any(b[0] <= x <= b[3] and b[1] <= y <= b[4] for b in room['tall'])
            if inside_tall:
                continue
            add((x, y, 0.0), (0, 0, 1), dx * dy, room['rho']['floor'])
            add((x, y, H), (0, 0, -1), dx * dy, room['rho']['ceiling'])
    for i in range(nx):
        for k in range(nz):
            x, z = (i + 0.5) * dx, (k + 0.5) * dz
            add((x, 0.0, z), (0, 1, 0), dx * dz, room['rho']['wall'])
            add((x, W, z), (0, -1, 0), dx * dz, room['rho']['wall'])
    for j in range(ny):
        for k in range(nz):
            y, z = (j + 0.5) * dy, (k + 0.5) * dz
            add((0.0, y, z), (1, 0, 0), dy * dz, room['rho']['wall'])
            add((L, y, z), (-1, 0, 0), dy * dz, room['rho']['wall'])
    # outer faces of tall partitions (bathroom) facing the main room
    for b in room['tall']:
        x0, y0, _, x1, y1, z1 = b
        for k in range(nz):
            z = (k + 0.5) * dz
            for t in np.arange(x0 + dx / 2, x1, dx):
                add((t, y0, z), (0, -1, 0), dx * dz, room['rho']['wall'])
            for t in np.arange(y0 + dy / 2, y1, dy):
                add((x1, t, z), (1, 0, 0), dy * dz, room['rho']['wall'])
    c = np.array([o[0] for o in out], float)
    n = np.array([o[1] for o in out], float)
    a = np.array([o[2] for o in out], float)
    r = np.array([o[3] for o in out], float)
    return c, n, a, r


def walkable(room, xy):
    x0, y0, x1, y1 = room['walk']
    ok = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    for b in room['keepout']:
        ok &= ~((xy[:, 0] > b[0] - 0.3) & (xy[:, 0] < b[3] + 0.3) &
                (xy[:, 1] > b[1] - 0.3) & (xy[:, 1] < b[4] + 0.3))
    return ok
