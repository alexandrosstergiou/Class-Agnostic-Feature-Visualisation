'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import glob
import cv2
import sys
import copy
import time
import math
import numpy as np
import torch
import scipy
from scipy import ndimage as nd
from skimage.draw import circle
from PIL import Image
from torch.optim import SGD, Adam
from torchvision import models
from imgaug import augmenters as iaa
from sys import getsizeof
import traceback
from decimal import Decimal

from torchviz import make_dot
import torch.nn.functional as F
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle, denoise_bilateral
from scipy.ndimage import zoom
from skimage import filters

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import robust_loss_pytorch.general


'''
---- S T A R T  O F  D I S T A N C E  F U N C T I O N S ----
'''

def l1_dist(input: torch.Tensor,target: torch.Tensor,reduce: str = "none") -> torch.Tensor:
    loss = torch.abs(input - target)
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

def l2_dist(input: torch.Tensor,target: torch.Tensor,reduce: str = "none") -> torch.Tensor:
    loss = torch.sqrt(torch.square(input - target))
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

def chebychev_dist(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    loss, _ = torch.max(torch.abs(input - target),-1)
    return loss

def canberra_dist(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    loss = torch.abs(input - target)/torch.abs(input) + torch.abs(target)
    return loss.sum(-1)

def minkowsky_dist(input: torch.Tensor,target: torch.Tensor,p: float = 1.,reduce: str = "none") -> torch.Tensor:
    loss = torch.pow(torch.pow(torch.abs(input - target),p),1./float(p))
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

'''
---- E N D  O F  D I S T A N C E  F U N C T I O N S ----
'''



'''
---- S T A R T  O F  P I E C E - W I S E  F U N C T I O N S  ----
'''

def logitdist_loss(input: torch.Tensor,target: torch.Tensor,p: float = 1.,reduce: str = "none") -> torch.Tensor:
    loss = minkowsky_dist(input,target,p)
    loss = torch.log(4*torch.exp(loss)/torch.square(1+torch.exp(loss)))
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

def huber_loss(input: torch.Tensor,target: torch.Tensor,delta: float = 1.,reduce: str = "none") -> torch.Tensor:
    loss = torch.abs(input - target)
    loss = torch.where(loss < delta, 0.5 * loss ** 2, loss * delta - (0.5 * delta ** 2))
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

def quantile_loss(input: torch.Tensor,target: torch.Tensor,q: float = 0.75,p: float = 2.,reduce: str = "none") -> torch.Tensor:
    loss = minkowsky_dist(input,target,p)
    loss = torch.where(loss >= 0, (1-loss)*q, - q * loss)
    if reduce == "mean":
        loss = loss.mean()
    elif reduce == "sum":
        loss = loss.sum()
    return loss

'''
---- E N D  O F  P I E C E - W I S E  F U N C T I O N S  ----
'''



"""
---- S T A R T  O F  F U N C T I O N  F O R M A T _ N P _ O U T P U T  ----

    [About]
        Converter to format WxHx3 and values in range of (0-255).

    [Args]
        - np_arr: Numpy array of shape 1xWxH or WxH or 3xWxH.

    [Returns]
        - np_arr: NUmpy array of shape WxHx3.
"""
def format_np_output(np_arr):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr
"""
---- E N D  O F  F U N C T I O N  F O R M A T _ N P _ O U T P U T  ----
"""

"""
---- S T A R T  O F  F U N C T I O N  S A V E _ I M A G E  ----

    [About]
        Save a numpy array to an image file.

    [Args]
        - imgs: List/Numpy array that contains images of shape WxHx3
        - path: String for the path to save location
        - iter: Integer for the iteration number to be added in file name

    [Returns]
        - None
"""
def save_image(imgs, basepath, iter):
    for i,im in enumerate(imgs):
        path = os.path.join(basepath,'cluster_{:02d}'.format(i+1),iter)
        if isinstance(im, (np.ndarray, np.generic)):
            im = format_np_output(im)
            im = Image.fromarray(im)
        im.save(path)
"""
---- E N D  O F  F U N C T I O N  S A V E _ I M A G E  ----
"""

"""
---- S T A R T  O F  P R E P R O C E S S _ I M A G E  ----

    [About]
        Converter for images from arrays to CNN-friendly format.

    [Args]
        - imgs: List/Numpy array containing strings of the filepaths
        - for the images to be loaded
        - resize_im: Boolean value for image resizing

    [Returns]
        - im_as_var: PyTorch tensor of shape [Bx3xWxH] with values
        between (0-1).
"""
def preprocess_image(imgs, resize_im=True):
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        new_imgs = []
        for im in imgs:
            im = cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
            new_imgs.append(im)
        imgs = np.asarray(new_imgs)
    im_as_arr = np.float32(imgs)
    im_as_arr = im_as_arr.transpose(0, 3, 1, 2)  # Convert array to B,C,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr[0]):
        im_as_arr[:,channel] /= 255
        im_as_arr[:,channel] -= mean[channel]
        im_as_arr[:,channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Convert to Pytorch variable
    im_as_var = im_as_ten.clone().detach().cuda().requires_grad_(True)
    return im_as_var
"""
---- E N D  O F  P R E P R O C E S S _ I M A G E  ----
"""


"""
---- S T A R T  O F  R E C R A T E _ I M A G E  ----

    [About]
        Reverse of `image_processing`. Converts images back to
        numpy arrays from tensors.

    [Args]
        - im_as_var: PyTorch tensor of shape [Bx3xHxW] corresponding
        to the image with values (0-1).

    [Returns]
        - recreated_im: Numpy array of shape [BxHxWx3] with values (0-225).
"""
def recreate_image(im_as_var):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = im_as_var.clone().detach().cpu().data.numpy()
    for c in range(3):
        recreated_im[:,c] /= reverse_std[c]
        recreated_im[:,c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(0, 2, 3, 1)
    return recreated_im
"""
---- E N D  O F  R E C R A T E _ I M A G E  ----
"""


"""
---- S T A R T  O F  C R E A T E _ C I R C U L A R _ M A S K  ----

    [About]
        Creates a circular mask with a Gaussian distribution.

    [Args]
        - h: Integer for the image height.
        - w: Integer for the image width.
        - centre: Tuple for the mask centre. If None will be
        the midle of the image.
        - radius: Integer for the circle radius. If None, it
        finds the smallest distance possible from the centre and
        the image borders.

    [Returns]

        - recreated_im: Numpy array of the masj with shape [HxW].
"""
def create_circular_mask(h, w, centre=None, radius=None):

    if centre is None: # use the middle of the image
        centre = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the centre and image borders
        radius = min(centre[0], centre[1], w-centre[0], h-centre[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_c = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_c <= radius
    return mask
"""
---- E N D  O F  C R E A T E _ C I R C U L A R _ M A S K  ----
"""


'''
===  S T A R T  O F  C L A S S  V I S G E N E R A T I O N ===

    [About]
        Function for creating image visualisations from selected CNN layers and channels.

    [Init Args]
        - features: A `torch.nn.Sequential` object containing the model to be visualised.
        - target_channels: The integer number of channels to be visualised.
        - clusters: The integer number of sub-sets/clusters to be used (i.e. facets).
        - octave: Dictionary of editable sub-parameters:
          + `scale`: Float for the scale to be used during interpolation w.r.t. the image.
          + `start_lr`: Float for the initial learning rate.
          + `end_lr`: Float for the final learning rate value (if same as initial there will be no change in the learning rate).
          Experiments up to this point only consider (`start_lr`>=`end_lr`).
          + `start_mask_sigma`: Integer, initial standard deviation value for the gradient mask.
          + `end_mask_sigma`: Integer, final standard deviation value for the gradient mask (if same as initial there will
          be no change in the gradient mask sigma). Experiments up to this point only consider (`start_mask_sigma`>=`end_mask_sigma`).
          + `start_sigma`: Float for initial Gaussian blur sigma value.
          + `end_sigma`: Float for final Gaussian blur sigma value
          + `start_denoise_weight`: Integer for initial denoising weight (small weights correspond to more denoising ~ smaller similarity to input)
          + `end_denoise_weight`: Integer for final denoising weight
          + `start_lambda`: Float for initial lambda value used for scaling the regularized layers.
          + `end_lambda`: Float for final lambda value for regularization. (`start_lambda==end_lambda` would correspond to no change in the regularization scale)
          + `window`: Integer for the size of the window area to be used for cropping.
          + `window_step`: Integer for cropping step.
          + `brightness_step`: Integer number of iterations after which a small value in the overall brightness is added.
        - img_name: String for the name of the folder containing all the images. This can for example correspond to a specific ImageNet class.
        - target: String for the target layer to be visualized. If unsure, use a loop to print all the modules:
        - penalty: String for the name of the layer to apply regularisation to.
        - iterations: Integer for the total number of iterations.
        - data_path: String for the full directory where the images to be used are. (do not include the specific `img_name` folder this is added at the end of
        the path during the class initialisation)

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - get_activation:
        - find_new_val:
        - generate:
'''
class VisGeneration():
    def __init__(self, model, target_top_n_features, num_clusters, target,penalty, octave, img_name='n03063599', iterations=2001, data_path='/ILSVRC2012/train/'):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.iterations = iterations
        self.epochs = 0
        self.octave = octave
        self.activations = {}
        self.model = model
        self.target = target
        self.penalty = penalty
        self.multigpu = False

        for name, module in self.model.named_modules():
            if (name == self.penalty):
                module.register_forward_hook(self.get_activation(name,'in'))
            if (name == self.target):
                module.register_forward_hook(self.get_activation(name,'out'))

        self.model.eval()
        self.initial_learning_rate = octave['start_lr']
        self.decrease_lr = octave['end_lr']
        self.ms_start = octave['start_mask_sigma']
        self.ms_end = octave['end_mask_sigma']

        # Lading target image
        images = []
        outputs = []
        dict_idx = {}
        cpu = torch.device("cpu")
        for i,image in enumerate(glob.glob(os.path.join(data_path,img_name,'*'))):
            try:
                image_example_tensor = cv2.imread(image)
                dict_idx[i]=image
                image_example_tensor = preprocess_image(np.expand_dims(image_example_tensor,axis=0),True)
                with torch.no_grad():
                    _ = self.model(image_example_tensor)
                output = self.activations[self.target+'_out'].clone().to(cpu).detach().requires_grad_(False)
                tensor_size = list(output.size())
                pooled_out = F.avg_pool2d(output.squeeze(0),(tensor_size[-2],tensor_size[-1])).squeeze(-1).squeeze(-1).detach().data.numpy()
                outputs.append(pooled_out)
                tmp = recreate_image(image_example_tensor).squeeze(0)
                images.append(tmp)
                print('Processed image {0:03d}'.format(i))
            except Exception as e:
                traceback.print_exc()
                print('Skipping image {0:03d}'.format(i))
                continue

        # Switch to DataParallel
        self.device_ids = [0,1]
        self.model = torch.nn.DataParallel(self.model, [0,1])
        self.multigpu = 2

        # Reduce dimensionality to 50 w/ PCA
        pca = PCA(n_components=50)
        pca.fit(outputs)
        reduced_outputs = pca.transform(outputs)

        # Create 2D Embeddings for clustering w/ tSNE
        tsne = TSNE(n_components=2)
        e2d_outputs = tsne.fit_transform(reduced_outputs)

        # Create clusters with KMeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(e2d_outputs)
        clusters_outputs = kmeans.predict(e2d_outputs)

        centres_idx = []
        centres = []
        distances = []
        print('Approximating closest clips ...')
        for c,cluster in enumerate(kmeans.cluster_centers_) :
            distance = kmeans.transform(e2d_outputs)[:, c]
            distances.append(distance)
            idx = np.argmin(distance)
            centres_idx.append(idx)
            centres.append(mpimg.imread(dict_idx[idx]))

        fig, ax = plt.subplots()
        for i,image in enumerate(centres):

            imagebox = OffsetImage(cv2.resize(image,dsize=(112,112),interpolation=cv2.INTER_NEAREST), zoom=0.25)
            ab = AnnotationBbox(imagebox, kmeans.cluster_centers_[i])
            ax.add_artist(ab)

        ax.scatter(e2d_outputs[:, 0], e2d_outputs[:, 1], c=clusters_outputs, s=50, cmap='mako',edgecolors='black')
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.75,edgecolors='black')

        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        plt.savefig('tsne_{:s}.png'.format(img_name),dpi=400,figsize=(12,10))

        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

        # find n=10 outputs close to each cluster centre
        clustered_images = {}
        for i in range(num_clusters):
            indices = np.argsort(distances[i])[::-1][:10]
            clustered_images[i] = [[images[indx] for indx in indices],[outputs[indx] for indx in indices], [distances[i][indx] for indx in indices]]

            # create directory
            if not os.path.exists('./generated/cluster_{:02d}'.format(i+1)):
                os.makedirs('./generated/cluster_{:02d}'.format(i+1))

        # Iterate over the pooled volume and find the indices that have a value larger than the threshold
        # Get all values above threshold
        self.target_channels = []
        self.rates = []
        for i in range(num_clusters):
            # Create soft-distance mask: e**(1/dist)/sum(e**(1/dist))
            mask = np.exp((1/np.asarray(clustered_images[i][2])))/np.sum(np.exp(1/np.asarray(clustered_images[i][2])),axis=0)
            sumout_i = sum(np.asarray(clustered_images[i][1])*np.expand_dims(mask,axis=(1)))
            values = [value for value in sumout_i]
            # accending order sort
            values.sort()
            # select last n values
            values = values[-target_top_n_features:]
            # find top n value indices in pooled tensor
            c_indices = [j for j, feat in enumerate(sumout_i) if feat in values]
            # get pooled volume sum for indices rates
            total = sum(values)
            rates = [(sumout_i[idx]/total) for idx in c_indices]

            message_m = 'Cluster {:0d} :: Generating images for channel indices: '.format(i)
            for c_i,r in zip(c_indices,rates):
                message_m += 'idx:{}'.format(c_i)
            print(message_m)

            self.target_channels.append(c_indices)
            self.rates.append(mask)

        # Generate a random image
        imgs = np.asarray([np.asarray(clustered_images[idx][0])*np.expand_dims(self.rates[idx],axis=(1,2,3)) for idx in range(num_clusters)])
        acts = np.asarray([np.asarray(clustered_images[idx][1])*np.expand_dims(self.rates[idx],axis=(1)) for idx in range(num_clusters)])

        sum_acts = np.asarray([np.sum(act,axis=0) for act in acts])
        self.target_maps = sum_acts.astype(np.float32)
        self.created_images = np.asarray([np.sum(img,axis=0) for img in imgs]) #np.uint8(np.full((3, 512, 512),117))
        self.created_images = self.created_images.transpose(0, 3, 1, 2).astype(np.uint8)

        save_image(self.created_images,'./generated/','iteration_0.jpg')




    def get_activation(self,name,mode):
        def hook(model, input, output):
            if mode == 'in':
                if self.multigpu:
                    self.activations[name+'_in_c'+str(input[0].get_device())] = input#.to(torch.device('cuda:0'))
                else:
                    self.activations[name+'_in'] = input
            else:
                if self.multigpu:
                    self.activations[name+'_out_c'+str(output[0].get_device())] = output#.to(torch.device('cuda:0'))
                else:
                    self.activations[name+'_out'] = output
        return hook

    def find_new_val(self,start,end,i,tot_i):
        a = (start-end) / (1-tot_i)
        b = (end-start*tot_i) / (1-tot_i)
        return (a*i)+b

    def generate(self,octave=None):

        _, _, h, w = self.created_images.shape

        if (octave is not None):
            self.octave = octave

        # resizing with interpolation
        images = nd.zoom(self.created_images, (1,1,self.octave['scale'],self.octave['scale']))
        random_crop = True
        _, _, imh, imw = images.shape
        total_loss = 0
        avg_loss = 0
        class_loss = 0
        tot_i = self.iterations

        print('\033[104m --- New cycle of iterations initialised --- \033[0m')

        start =1
        if (self.epochs != 0):
            start = self.epochs
            tot_i += start

        for i in range(start, tot_i+1):

            start_time = time.time()

            # Learning rate decrease
            lrate = self.find_new_val(self.octave['start_lr'],self.octave['end_lr'],i,tot_i)

            # Sigma decrease
            mask_sigma = self.find_new_val(self.octave['start_mask_sigma'],self.octave['end_mask_sigma'],i,tot_i)

            # Update blur sigma
            sigma = self.find_new_val(self.octave['start_sigma'],self.octave['end_sigma'],i,tot_i)

            # Update denoise weight
            denoise_weight = self.find_new_val(self.octave['start_denoise_weight'],self.octave['end_denoise_weight'],i,tot_i)

            # Update L1 lambda
            l1_lambda = self.find_new_val(self.octave['start_lambda'],self.octave['end_lambda'],i,tot_i)

            if imw > w:
                if random_crop:
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x*self.octave['window'], 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y*self.octave['window'], 1)
                    oy = int(np.clip(oy,0,imh-h))
                else:
                    ox = int((imw-w)/2.)
                    oy = int((imh-h)/2.)

                if (i%self.octave['window_step'] == 0):
                    self.created_images = images[:,:,oy:oy+h,ox:ox+w]

            else:
                ox = 0
                oy = 0


            # Create masks
            mask1 = np.ones((self.created_images.shape[-2],self.created_images.shape[-1]), dtype=np.float32)
            for y in range(mask1.shape[0]):
                for x in range(mask1.shape[1]):
                    cx = mask1.shape[1]//2
                    cy = mask1.shape[0]//2
                    val = math.sqrt((abs(y-cy)**2)+(abs(x-cx)**2))
                    mask1[y,x] = 1/(mask_sigma * math.sqrt(2*math.pi)) * sys.float_info.epsilon**(.5 * ((val-1)/mask_sigma)**2)

            # Normalise (0-1)
            mask1 = (mask1-mask1.min())/(mask1.max()-mask1.min())
            mask2 = abs(1-mask1)


            # Blur image
            if (i%10 ==0):
                blurred = iaa.GaussianBlur((sigma))
                self.created_images = self.created_images.transpose(0,2,3,1).astype(np.float32)
                blurred_image = blurred(images=self.created_images).astype(np.float32)

                # Combine new image and previous image
                blurred_image *= np.asarray([mask1,mask1,mask1]).transpose(1,2,0)
                self.created_images *= np.asarray([mask2,mask2,mask2]).transpose(1,2,0).astype(np.float32)
                self.created_images += blurred_image

                # Add brightness
                if(i%self.octave['brightness_step'] == 0 and i>0):
                    self.created_images += 5
                    self.created_images = np.clip(self.created_images, 0, 255)

                self.created_images = self.created_images.transpose(0,3,1,2).astype(np.uint8)


            images[:,:,oy:oy+h,ox:ox+w] = self.created_images

            # Process image and return variable
            self.processed_images = preprocess_image(self.created_images.transpose(0,2,3,1), False)

            # Define loss function
            adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                            num_dims = self.target_maps.shape[1],
                            float_dtype = np.float32,
                            device = 'cuda:0')

            # Define optimizer for the image
            optimizer = SGD([self.processed_images]+list(adaptive.parameters()), lr=lrate,momentum=0.9,nesterov=True)

            # Forward
            torch.cuda.empty_cache()
            _ = self.model(self.processed_images)

            if self.multigpu:
                #print(self.activations.keys())
                out = torch.cat([ self.activations[self.target+'_out_c'+str(id)].to(torch.device('cuda:0')) for id in self.device_ids ],dim=0)
            else:
                out = self.activations[self.target+'_out']

            # Esure shape of [B,C]
            tensor_shape = list(out.size())
            if (len(tensor_shape) > 2):
                if (tensor_shape[-1] > 1):
                    out = F.avg_pool2d(out, kernel_size=(out.shape[-2],out.shape[-1])).squeeze(-1).squeeze(-1)
                else:
                    out = out.squeeze(-1).squeeze(-1)

            # Get target activations difference
            act_loss = torch.mean(adaptive.lossfun((out - torch.tensor(self.target_maps, requires_grad=False).to(torch.device('cuda:0')))))
            #act_loss = - quantile_loss(out,torch.tensor(self.target_maps, requires_grad=False).to(torch.device('cuda:0'))).mean(-1)

            if isinstance(class_loss,int):
                prev_class_loss = 0
            else:
                prev_class_loss = copy.deepcopy(class_loss.item())

            # Calculate dot product
            class_loss = -1 * torch.sum(out[:,self.target_channels] * torch.tensor(self.target_maps[:,self.target_channels], requires_grad=False).to(torch.device('cuda:0')),dim=-1)

            if self.multigpu:
                p_layer = torch.cat([ self.activations[self.penalty+'_in_c'+str(id)][0].to(torch.device('cuda:0')) for id in self.device_ids ],dim=0)
            else:
                p_layer = self.activations[self.penalty+'_in'][0]

            l1_penalty = l1_lambda * torch.norm(p_layer,p=1).sum(-1).squeeze(-1).sum(-1).squeeze(-1).mean(-1)
            class_loss += l1_penalty
            class_loss += act_loss
            class_loss = class_loss.sum()

            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()

            self.processed_images.grad *= torch.from_numpy(mask1).to(torch.device('cuda:0'))

            # Update image
            optimizer.step()
            prev_avg_loss = copy.deepcopy(avg_loss)
            avg_loss += class_loss.item()

            # Build string for printing loss change

            # -- CLASS LOSS STRING
            # Large positive increase
            if (abs(class_loss) - abs(prev_class_loss) > 100):
                loss_string = '-- Loss ['+'\033[92m' +'{0:.2e}'.format(Decimal(class_loss.item()))+ '\033[0m'+'/'
            # Smaller positive increase
            elif (abs(class_loss) - abs(prev_class_loss) > 0):
                loss_string = '-- Loss ['+'\033[93m' +'{0:.2e}'.format(Decimal(class_loss.item()))+ '\033[0m'+'/'
            # Negative decrease
            else:
                loss_string = '-- Loss ['+'\033[91m' +'{0:.2e}'.format(Decimal(class_loss.item()))+ '\033[0m'+'/'

            # -- AVRG LOSS STRING
            # Large positive increase
            if (abs(avg_loss) - abs(prev_avg_loss) > 100):
                loss_string += '\033[92m' +'{0:.2e}]'.format(Decimal(avg_loss/5))+ '\033[0m'
            # Smaller positive increase
            elif (abs(avg_loss) - abs(prev_avg_loss) > 0):
                loss_string += '\033[93m' +'{0:.2e}]'.format(Decimal(avg_loss/5))+ '\033[0m'
            # Negative decrease
            else:
                loss_string += '\033[91m' +'{0:.2e}]'.format(Decimal(avg_loss/5))+ '\033[0m'

            print('\033[94m' +'Iteration: [{0:05d}/{1:05d}]  '.format(i,tot_i)+'\033[0m'+
                    loss_string+
                    ' -- Mask [\033[91mv\033[0m/\033[93m-\033[0m/\033[92m^\033[0m] [\033[91m{0:.3f}\033[0m/\033[93m{1:.3f}\033[0m/\033[92m{2:.3f}\033[0m] -- Sigma (\033[96mblur/denoise\033[0m) [\033[96m{3:.4f}/{4:.4f}\033[0m] -- lr [{5:.2e}] -- time {6:.2f}s.'
                    .format(
                    mask1.min(),
                    mask1.mean(),
                    mask1.max(),
                    sigma,
                    denoise_weight,
                    Decimal(lrate),
                    time.time()-start_time))

            total_loss += avg_loss
            avg_loss = 0

            # Recreate image
            self.created_images = recreate_image(self.processed_images)

            # Denoise image

            if (i%10 == 0):
                denoised_imgs = []
                for created_image in self.created_images:
                    denoised_imgs.append(denoise_tv_bregman(created_image, weight=denoise_weight, max_iter=100, eps=1e-3).astype(np.float32))
                denoised_imgs = np.asarray(denoised_imgs)
                denoised_imgs *= 255

                # apply masks
                self.created_images = self.created_images.astype(np.float32)
                denoised_imgs *= np.asarray([mask1,mask1,mask1]).transpose(1,2,0)
                self.created_images *= np.asarray([mask2,mask2,mask2]).transpose(1,2,0).astype(np.float32)

                self.created_images += denoised_imgs
                self.created_images = self.created_images.astype(np.uint8)

                # Save image
                im_path = 'iteration_{:04d}.jpg'.format(i)

                if (i % 200 == 0):
                    save_image(self.created_images,'./generated', im_path)

            self.created_images = self.created_images.transpose(0,3,1,2)

        self.epochs = tot_i
        return avg_loss/tot_i
'''
===  E N D  O F  C L A S S  V I S G E N E R A T I O N ===
'''



if __name__ == '__main__':
    target_channels = 50
    clusters = 10
    penalty_layer = '7.2.conv3'
    target_layer = '7.2.relu'

    pretrained_model = models.resnet152(pretrained=True).cuda()
    features = torch.nn.Sequential(*list(pretrained_model.children())[:-2])

    # Uncomment this for user-defined paramaters

    vis = VisGeneration(features, target_channels, clusters,
             octave={
                'scale':1.2,
                'start_lr':5e-1,'end_lr':8e-2,
                'start_mask_sigma':420,'end_mask_sigma':340,
                'start_sigma':0.9,'end_sigma':0.2,
                'start_denoise_weight':2,'end_denoise_weight': 8,
                'start_lambda':1e-4,'end_lambda':5e-4,
                'window':0.4,
                'window_step':400,
                'brightness_step':50},
            img_name='n02169497',
            target = target_layer,
            penalty = penalty_layer,
            iterations=2001)

    vis.generate()
