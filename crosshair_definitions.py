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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

###############################################################################
'''
takes a line in Hesse-form (r,theta) and gives back two points on the line in 
xy-coordinates outside of the image borders
INPUT
radius:radius of Hesse-form
theta: angle theta of Hesse-form
OUTPUT
x1: x-coordinate of first point
y1: y-coordinate of first point
x2: x-coordinate of second point
y2: y-coordinate of second point
'''
###############################################################################
def hesse_to_xy(radius,theta):
    ln = 6000
    if abs(theta) < 1e-4: #avoid theta=0
        x1 = int(radius)
        y1 = ln
        x2 = int(radius)
        y2 = -ln   
    else:
        m = -1.0/np.tan(theta)
        b = radius/np.sin(theta)
        x1=ln
        y1 = int(m*x1+b)
        x2 = -1*int(ln)
        y2 = int(m*x2+b)
    return ((x1),(y1)),((x2),(y2))

###############################################################################
'''
takes an array of lines and finds the 'number_of_lines' best fitting lines
INPUT
lines: nx2 array of n lines in Hesse normal form
number_of_lines: number of lines which are needed (4 per cross)
space: two lines need to have more than 'space' pixels distance from each other
to be considered 
OUTPUT
valid_lines: number_of_linesx2 array of lines in Hesse normal form
'''
###############################################################################
def find_fittest_lines4(lines,number_of_lines,ds,image):
    valid_lines = np.zeros_like(lines[0:number_of_lines,:]) #container for valid lines
    if lines[0,0]<0:
        valid_lines[0,0]=abs(lines[0,0])
        valid_lines[0,1]=lines[0,1]-np.pi     #first entry
    else:
        valid_lines[0,:]=lines[0,:]
    j= 0 #runs through 'lines'
    for i in range(number_of_lines-1):
        valid_lines[i+1]=i+1
        count = 1    #abbort criterion for while loop
        while count == 1:
            count = 0 #reset abort criterion
            j+=1
            take = lines[j,:]
            if take[0]<0:
                take[0] = abs(take[0])
                take[1] = take[1]-np.pi
            for k in range(number_of_lines): #take line j and check if within 'space' pixels of line k
                if  lines_are_close(take,valid_lines[k,:],ds,image):
                    count = 1#if line j is too close, run loop again with line j=j+1
        valid_lines[i+1,:]=take   #keep line j if it has nececarry distance 
    return valid_lines

###############################################################################
'''
takes an array of lines in Hesse normal form and sorts according to angle 
(smallest angle first)
INPUT
lines_input: nx2 array of n lines in Hesse normal form (unsorted)
OUTPUT
sorted_lines: nx2 array of n lines sorted according to angle
'''
###############################################################################
def sort_lines(lines_input):
    for i in range(lines_input.shape[0]): #check for negative radii
        if lines_input[i,0]<0:
            lines_input[i,0]=abs(lines_input[i,0])
            lines_input[i,1]=lines_input[i,1]-np.pi
    sorted_lines = np.zeros_like(lines_input)
    sort = np.argsort(lines_input[:,1])
    for i in range(lines_input.shape[0]):
        sorted_lines[i]=lines_input[sort[i]]
    return sorted_lines

###############################################################################
'''
finds center coordinate of FOUR lines
INPUT
sorted_lines:4x2 array of lines in Hesse normal form, sorted according to angle 
(smallest angle first)
OUTPUT
xy coordinate of center'''
###############################################################################
def find_center(sorted_lines):
    a = np.array([1.1,1.1])
    a[0]=(0.5*(sorted_lines[0,0]+sorted_lines[1,0]))
    a[1]=(0.5*(sorted_lines[0,1]+sorted_lines[1,1]))
    b = np.array([0.0,0.0])
    b[0]=(0.5*(sorted_lines[2,0]+sorted_lines[3,0]))
    b[1]=(0.5*(sorted_lines[2,1]+sorted_lines[3,1]))
    r_a = a[0]
    th_a = a[1]+1e-6    #### avoid th_a=0
    r_b = b[0]
    th_b = b[1]+1e-6    #### avoid th_b=0
    cx = 1.0/(np.tan(th_a)-np.tan(th_b))*(r_b*np.sin(th_a)-r_a*np.sin(th_b))/(np.cos(th_b)*np.cos(th_a))
    cy = 1.0/np.sin(th_a)*(r_a-cx*np.cos(th_a))
    return cx,cy

###############################################################################
'''
performs Hough Line Transform
INPUT
edges: image containing contours
r_res: radial resolution of HLT
theta_res: angular resolution of HLT
v_threshold: only lines with vote above 'v_threshold' are considered
OUTPUT
lines: nx2 array of all the lines with at least a vote of 'v_treshold'
'''
###############################################################################
def do_HoughLines(edges,r_res,theta_res,v_threshold):
    lines1 = cv2.HoughLines(edges,r_res,theta_res,v_threshold)
    lines = lines1[:,0,:]
    return lines

###############################################################################
'''
determines whether two lines represent the same edge
INPUT

OUTPUT
True: the two lines are within 'ds' pixels of each other
False: the two lines are farther apart
'''
###############################################################################
def lines_are_close(line_1,line_2,ds,image):
    if abs(line_1[0]-line_2[0]) < ds:
        return True
    else:
        return False
    
###############################################################################
'''
converts the image to a binarey image to be fed to the HLT algorithm
INPUT
im: nxmx3 array, RGB image
thresh: threshold value between black and white
OUTPUT
edges: binary image as nxmx3 array
'''
###############################################################################
def rgb2edges_Canny(im,thresh):
    im2 = np.zeros_like(im)
    grey_int255 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY )
    grey_int255[grey_int255>thresh]=255
    grey_int255[grey_int255<=thresh]=0
    im2[:,:,0]=grey_int255
    im2[:,:,1]=grey_int255
    im2[:,:,2]=grey_int255
    edges = cv2.Canny(im2,80,70)
    return edges

###############################################################################
'''
draws lines
INPUT
nx2 array of n lines in Hesse normal form 
'''
###############################################################################
def drawlines(lines_array,image,thickness):
    for i in range(lines_array.shape[0]):
        p1,p2 = hesse_to_xy(lines_array[i,0],lines_array[i,1])
        cv2.line(image,p1,p2,(0,0,255),thickness)
    return 0
###############################################################################
'''
splits an 8x2 array of lines in Hesse normal for into two 4x2 arrays sorted by angle
INPUT
8x2 array of 8 lines in Hesse normal form in format (radius, theta)
OUTPUT
chair1:4x2 array of four lines in Hesse normal form (contains line with smallest angle)
chair2: 4x2 array of four lines in Hesse normal form (contains line with largest angle)
'''
###############################################################################
def separate_crosshairs(lines):
    nr = int(lines.shape[0]/4)
    chair1 = np.zeros_like(lines[0:2*nr])
    chair2 = np.zeros_like(lines[0:2*nr])    
    lines_sorted = sort_lines(lines)
    for i in range(nr):
        chair1[2*i:2*i+2,:] = lines_sorted[4*i:4*i+2,:]
        chair2[2*i:2*i+2,:] = lines_sorted[4*i+2:4*i+4,:]
    return chair1,chair2

###############################################################################
'''
finds two crosshair in image 'im' and returns their center coordinates with HLT
INPUT
im: nxmx3 array, RGB image
OUTPUT
fit_lines: 8x2 array of lines in Hesse normal form (Radius, Theta), sorted by
theta, starting with smallest theta
(cx1,cy1): xy center coordinates of crosshair 1
(cx1,cy1): xy center coordiantes of crosshair 2 
'''
###############################################################################
def find2Xhair(im):
    im_thres = 128 #luminance threshold
    edges = rgb2edges_Canny(im,im_thres)
    # find crosshairs ########################################
    r_res = 0.5 #radial resolution
    theta_res = np.pi/3000 #angular resolution
    v_threshold = 300 #number of votes to be considered 
    lines = do_HoughLines(edges,r_res,theta_res,v_threshold) 
    # find fittest lines ####################################
    number_of_lines = 8 #4 for single crosshair / 8 for two crosshairs
    space = 10 #pixels (10>= for thick crosshair)
    ds=20
    fit_lines = find_fittest_lines4(lines,number_of_lines,ds,im)#only pick the n best fitting lines (unsorted)
    chair1,chair2=separate_crosshairs(fit_lines)
    # find center ####################################
    cx1,cy1 = find_center(chair1) 
    cx2,cy2 = find_center(chair2) 
    return fit_lines,(cx1,cy1),(cx2,cy2)
    
