from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy import ndimage
import cv2
from skimage.color import rgb2gray
from astropy.convolution import Gaussian2DKernel, convolve



gauss_response = lambda x, x0, sigma, A: A*0.5*(1 + erf((x-x0)/(sigma**2*np.sqrt(2))))
gauss_response_inv = lambda x, x0, sigma, A: 1*A-gauss_response(x, x0, sigma, A)


def fit_gaussian_response(x, y, inverse=False):
    if inverse:
        p, pcov = curve_fit(gauss_response_inv,x , y, p0=[x[np.argmax(y)], 1, np.max(y)])
    else:
        p, pcov = curve_fit(gauss_response,x , y, p0=[x[np.argmax(y)], 1, np.max(y)])
    return p

def line_intersection(lx, ly):
    y = (lx[0]*ly[1]+lx[1])/(lx[0]*ly[0]+1)
    x = (2*ly[0]*lx[0]*ly[1]+ly[0]*lx[1]+ly[1])/(lx[0]*ly[0]+1)
    return x, y

def get_first_point(grad):
    xmax = np.argmax(grad)
    xmin = np.argmin(grad)

    if np.abs(xmax-xmin)<200:
        return xmin
    else:
        #set respective regions to 0
        if np.abs(xmax-len(grad)/2) < np.abs(xmin-len(grad)):
            grad[xmin-15:] = 0
        else:
            grad[:xmax+15] = 0
        return get_first_point(grad)


def proccess_cut_img(img, xr = None, yr=None, thresh = 1):
    if xr is not None:
        img = img[xr[0]:xr[1], :]
    if yr is not None:
        img = img[:,yr[0]:yr[1]]
    if thresh is not None:
        img = 1 - img
        img[img>thresh] = 1
        img = 1 - img
    return img
    
def edge_points(img, dx=50, dy=50, step=50, initX=200, initY=200):
    # first points for x and y direction
    cutx = img[initX, :]
    cuty = img[:, initY]
    x0 = get_first_point(np.gradient(cutx))
    y0 = get_first_point(np.gradient(cuty))
    #print(x0, y0)
    
    cutx = np.max(cutx) - cutx
    cuty = np.max(cuty) - cuty
    xdata = np.arange(x0-int(0.5*dx), x0+int(0.5*dx), 1)
    ydata = np.arange(y0-dy, y0+dy, 1)
    px = fit_gaussian_response(xdata, cutx[xdata])
    py = fit_gaussian_response(ydata, cuty[ydata])
    #print(px[0], py[0])
    cutoff_y1 = int(px[0]) # first edge point in x
    cutoff_x1 = int(py[0]) # first edge point in y
    sigmax, sigmay = px[1], py[1] # sigma ~ defocus

    edges_x = {"left":[], "right":[], "sigma_left":[], "sigma_right":[], "yi":[]}
    edges_y = {"left":[], "right":[], "sigma_left":[], "sigma_right":[], "xi":[]}

    xdata = np.arange(cutoff_y1+dx, cutoff_y1+2*dx)
    ydata = np.arange(cutoff_x1+dy, cutoff_x1+2*dy)
    px = fit_gaussian_response(xdata, cutx[xdata], inverse=True)
    py = fit_gaussian_response(ydata, cuty[ydata], inverse=True)

    cutoff_y2 = int(px[0]) # first edge point in x
    cutoff_x2 = int(py[0]) # first edge point in y

    pointsx = np.concatenate([np.arange(np.max([0,cutoff_x1-600]), cutoff_x1, step),
                              np.arange(cutoff_x2+step, np.min([len(cuty),cutoff_x1+600]), step)])
    pointsy = np.concatenate([np.arange(np.max([0,cutoff_y1-600]), cutoff_y1, step),
                              np.arange(cutoff_y2+step, np.min([len(cutx),cutoff_y1+600]), step)])
    #print('initial points', cutoff_y2, cutoff_x2)
    for yi in pointsx:

        cutx = img[yi,:]
        x0 = cutoff_y1
        cutx = np.max(cutx) - cutx
        xdata = np.arange(x0-dx, x0+dx)
        try:
            px = fit_gaussian_response(xdata, cutx[xdata])
        except RuntimeError:
            print("RuntimeError for yi %i"%yi)
            continue
        if len(cutx) < px[0] or px[0]< 0:
            print("Failed to find left edge for yi=%i"%yi)
            continue


        xdata = np.arange(int(px[0])+dx, int(px[0])+3*dx)
        try:
            px2 = fit_gaussian_response(xdata, cutx[xdata], inverse=True)
        except RuntimeError:
            print("RuntimeError for yi %i"%yi)
            continue
        if len(cutx) < px2[0] or px2[0]< 0:
            print("Failed to find right edge for yi=%i"%yi)
            continue
        
        edges_x["left"].append(px[0])
        edges_x["sigma_left"].append(px[1])

        edges_x["right"].append(px2[0])
        edges_x["sigma_right"].append(px2[1])
        edges_x["yi"].append(yi)


    for xi in pointsy:
        cuty = img[:,xi]
        y0 = cutoff_x1
        cuty = np.max(cuty) - cuty
        ydata = np.arange(y0-dy, y0+dy)
        try:
            py = fit_gaussian_response(ydata, cuty[ydata])
        except RuntimeError:
            print("RuntimeError for xi %i"%xi)
            continue
        if len(cuty) < py[0] or py[0]< 0:
            print("Failed to find left edge for xi=%i"%xi)
            continue

        ydata = np.arange(int(py[0])+dy, int(py[0])+3*dy)
        try:
            py2 = fit_gaussian_response(ydata, cuty[ydata], inverse=True)
        except RuntimeError:
            print("RuntimeError for xi %i"%xi)
            continue
        if len(cuty) < py2[0] or py2[0]< 0:
            print("Failed to right find edge for xi=%i"%xi)
            continue
        edges_y["left"].append(py[0])
        edges_y["sigma_left"].append(py[1])

        edges_y["right"].append(py2[0])
        edges_y["sigma_right"].append(py2[1])
        edges_y["xi"].append(xi)
    return edges_x, edges_y

def find_crosshair_center(img, step=50, **kwargs):
    ny, nx = np.shape(img)
    ex, ey = edge_points(img, step=50, **kwargs)

    lx1 = np.polyfit(ex["yi"], ex["left"], deg=1)
    lx2 = np.polyfit(ex["yi"], ex["right"], deg=1)
    zx = np.arange(0,nx, 0.1)

    ly1 = np.polyfit(ey["xi"], ey["left"], deg=1)
    ly2 = np.polyfit(ey["xi"], ey["right"], deg=1)
    zy =  np.arange(0,ny, 0.1)

    p1x, p1y = line_intersection(lx1, ly1)
    p2x, p2y = line_intersection(lx1, ly2)
    p3x, p3y = line_intersection(lx2, ly2)
    p4x, p4y = line_intersection(lx2, ly1)

    m1x = (p1x+p2x)/2
    m2x = (p3x+p4x)/2
    cx = (m1x+m2x)/2

    m1y = (p1y+p2y)/2
    m2y = (p3y+p4y)/2
    cy = (m1y+m2y)/2

    defocusx = np.mean(np.concatenate([ex["sigma_left"], ex["sigma_right"]]))
    defocusy = np.mean(np.concatenate([ey["sigma_left"], ey["sigma_right"]]))

    return cx, cy, zx, np.poly1d(lx1)(zx), np.poly1d(lx2)(zx), zy, np.poly1d(ly1)(zy),np.poly1d(ly2)(zy),defocusx, defocusy

def preprocess(data_in, thres=0.2):
    #d = rgb2gray(data_in)
    d = data_in
    edge_horizont = ndimage.sobel(d, 0)
    edge_vertical = ndimage.sobel(d, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    """magnitude[magnitude<thres]=0
    magnitude[magnitude>0] =1"""
    return magnitude


def find_edges1d(d, threshold):
    bind = np.zeros_like(d)
    bind[d>threshold] = 1
    first = np.argmax(bind)
    last = len(d)-np.argmax(np.flip(bind,axis=0))
    #plt.plot(np.arange(len(d)), d, first, threshold, 'ro', last, threshold, 'ro')
    return first, last, (first+last)/2


def trace_center_lines(img,theshold,step):
    """img[img<.2]=0
    img[img>0] =1"""
    edges_x = []
    centers_x = []
    for x in np.arange(0, len(img[:,0]), step):
        img_x = img[x,:]
        f, l, m = find_edges1d(img_x, theshold)
        edges_x.append([f,l])
        centers_x.append(m)

    edges_y = []
    centers_y = []
    for y in np.arange(0, len(img[0,:]), step):
        img_y = img[:,y]
        f, l, m = find_edges1d(img_y, theshold)
        edges_y.append([f,l])
        centers_y.append(m)

    plt.plot(centers_x,np.arange(0, len(img[:,0]), step),"rx",
             np.arange(0, len(img[0,:]), step), centers_y,"ro")

    return centers_x, centers_y, edges_x, edges_y, np.arange(0, len(img[0,:]), step)


def CrossHairKernel2D(N, w):
    if N%2 == 0:
        N+=1
    k = np.ones((N,N))*(-10)
    m = round(N/2)
    k[-w+m:m+w,:] = 1
    k[:,-w+m:m+w] = 1
    k[-w+m:m+w,-w+m:m+w] += 100*Gaussian2DKernel(3, x_size=2*w, y_size=2*w).array
    return k


def find_center2D(data_in, kernel_size):
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # we use 'valid' which means we do not add zero padding to our image
    edges = scipy.signal.convolve2d(data_in, kernel, 'valid')
    edges[np.abs(edges)<1] = 0
    edges[np.abs(edges)>=1] = 1

    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    edges_averaged =  scipy.signal.convolve2d(edges, kernel, 'valid')
    img_conv = convolve(edges_averaged,CrossHairKernel2D(71,25))
    plt.imshow(img_conv, interpolation='nearest')
