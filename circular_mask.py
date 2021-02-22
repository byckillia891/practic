import numpy as np
import matplotlib.pyplot as plt

def incirc(img, centers, r, value = 255, inside=True):
    if inside:
        for cen in centers:
            if (cen[0]<=0 or cen[1]<=0):
                continue
            pcirc(img, cen, r, value, True)
    else:
        mask = np.zeros(img.shape, dtype = np.uint8)
        for cen in centers:
            if (cen[0]<=0 or cen[1]<=0):
                continue
            pcirc(mask, cen, r, 255, True)
        img = img & (mask)

def pcirc(img, cen=None, r=None, value = 255, inside=True):
    mask  = gmask(img, cen, r)
    if inside:
        img[mask==1] = value
    else:
        raise Exception ('doesnt work')
    return mask

def gmask(img, cen=None, r=None):
    h, w = img.shape[0], img.shape[1]
    if cen is None:
        cen = [int(w / 2), int(h / 2)]
    if r is None:
        r = min(cen[0], cen[1], w - cen[0], h - cen[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cen[0]) ** 2 + (Y - cen[1]) ** 2)
    mask = dist_from_center <= r
    return mask

if __name__ == '__main__':
    sz = (128,1024)
    img = np.ones(sz, dtype=np.uint8) * (np.random.rand(sz[0],sz[1] )*255)
    img1 = np.copy(img)
    for i in range (7):
        pcirc(img1, (128 * (i + 1), 64), 32, i * (255 - 32), inside=True)
    fig, ax = plt.subplots(nrows=2, ncols=1 )
    ax[0].imshow(img, interpolation='none', cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(img1, interpolation='none', cmap='gray', vmin=0, vmax=255)
    plt.show()

