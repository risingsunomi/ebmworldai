"""
Displays an energy map of the contours of an image after
going through alexnet and NCE loss
"""
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.distributions as D
import torch.nn.functional as tnf
import matplotlib.pyplot as plt

from captures import Captures
from alexnet_features import AlexNetFeatures
from ebm import EBM

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.WARNING) # git rid of PIL logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # get rid of matplotlib.font_manager logging
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING) # get rid of matplotlib.pyplot
logger = logging.getLogger('desp_img_energy')

def preprocess_image(image):
    # using imagenet mean and std
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def visualize_img_energy_map_3d(
    image,
    temperature=1.0
):
    cframe = image
    cframe = torch.from_numpy(cframe).float()

    # Initialize the energy map
    energy_map = np.zeros((image.shape[0], image.shape[1]))
    height, width = energy_map.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the NCE loss
    nce_loss = tnf.binary_cross_entropy_with_logits(
        cframe, torch.ones_like(cframe), reduction='none'
    )

    # Reshape the nce_loss tensor to match the shape of the input image
    nce_loss = nce_loss.view(image.shape[2], image.shape[0], image.shape[1])
    nce_loss = nce_loss.permute(1, 2, 0).contiguous()
    nce_loss = nce_loss.view(image.shape[0], image.shape[1], -1)

    # Calculate the energy map
    energy_map = -temperature * nce_loss.numpy().squeeze()

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Select only one channel from the energy map
    if energy_map.ndim == 3:
        energy_map = energy_map[:, :, 0]

    # Plot the energy map as a 3D surface
    ax.plot_surface(X, Y, energy_map, cmap='jet')

    # Set labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Graph')

    # Convert the Matplotlib figure to an OpenCV image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Close the Matplotlib figure to free up memory
    plt.close(fig)

    return img


def visualize_contour_energy_3d(
    energy_map
):
    """
    3D graph display of the contour energies
    """
    height, width, depth = energy_map.shape

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for the X, Y, and Z coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = np.zeros_like(X)

    # Plot the 3D surface for each depth level
    for i in range(depth):
        Z = energy_map[:, :, i]
        ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.8)

    # Set labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.set_title('Energy Tensor')

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)

    # Convert the Matplotlib figure to an OpenCV image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Close the Matplotlib figure to free up memory
    plt.close(fig)

    return img

def main():
    # load energy based model
    ebm = EBM()
    
    # load alexnet only features
    anf = AlexNetFeatures()

    # Load frames
    caps = Captures()
    cap_frames = caps.get_all_pframes(limit=1000)
    frame_cnt = 0
    for cap_frame in cap_frames:
        logger.info(f"Processing Frame: {frame_cnt}")
        # convert blob to image array
        cap_txt = cap_frame[1]
        cap_array = caps.convert_array(cap_txt)

        # Preprocess the image
        preprocessed_image = preprocess_image(cap_array)
        preprocessed_image = preprocessed_image.unsqueeze(0)  # Add batch dimension

        # get alexnet features from image
        an_features = anf(preprocessed_image)
        
        # load gaussian distribution
        zeros_tensor = torch.zeros((an_features.shape))
        eye_tensor = torch.eye(an_features.shape[-1])

        noise = D.MultivariateNormal(zeros_tensor, eye_tensor)

        # generate noise from image for energy calc
        gen_noise = noise.sample()

        # get energy
        feature_energy, _, feature_tensor_energy = ebm.nce_loss(noise, an_features, gen_noise)

        logging.info(f"features_energy: {feature_energy}\n")

        # resize features energy tensor
        feature_tensor_energy = cv2.resize(
            feature_tensor_energy.squeeze().cpu().detach().numpy(),
            (cap_array.shape[1], cap_array.shape[0])
        )

        ftd_img = visualize_img_energy_map_3d(cap_array, 0.1)
        td_img = visualize_contour_energy_3d(feature_tensor_energy)

        # # Display the original image and the energy map
        # if cap_frame[2] != "":
        #     print(f"\ncaption: {cap_frame[2]}\n")
        
        
        cv2.imshow("Original Frame", cap_array)
        cv2.imshow("3D Energy Map of Frame", ftd_img)
        cv2.imshow("3D Energy Map of Contours", td_img)

        frame_cnt += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()