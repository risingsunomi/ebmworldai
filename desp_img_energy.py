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


def visualize_tensor_3d(in_tensor, title):
    """
    3D graph display of tensors
    """
    
    height, width, depth = in_tensor.shape

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for the X, Y, and Z coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = np.zeros_like(X)

    # Plot the 3D surface for each depth level
    for i in range(depth):
        Z = in_tensor[:, :, i]
        ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.8)

    # Set labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.set_title(title)

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

def draw_function_image(points, title, x_title, y_title):
    """
    Draw a function from a list of (x, y) tuples using Matplotlib and return the image.

    Args:
        points: A list of (x, y) tuples representing the points of the function.

    Returns:
        The image of the drawn function.
    """
    # Extract x and y values from the points
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the function
    ax.plot(x_values, y_values, linewidth=2, color='blue')

    # Set labels and title
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)

    # Adjust the plot limits
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(y_values), max(y_values))

    # Convert the Matplotlib figure to an OpenCV image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Close the Matplotlib figure to free up memory
    plt.close(fig)

    return img

def main():
    # load alexnet only features
    anf = AlexNetFeatures()

    # energies list
    frame_energy_list = []
    feature_energy_list = []

    # Load frames
    caps = Captures()
    cap_frames = caps.get_all_pframes(limit=2000)
    frame_cnt = 0
    for cap_frame in cap_frames:
        logger.info(f"Processing Frame: {frame_cnt}")
        # convert blob to image array
        cap_txt = cap_frame[1]
        cap_array = caps.convert_array(cap_txt)

        # Preprocess the frame
        proc_frame = preprocess_image(cap_array)
        proc_frame = proc_frame.unsqueeze(0)  # Add batch dimension

        # --------- frame image energy ---------- #
        logging.info(f"proc_frame.shape {proc_frame.shape}")

        # load gaussian distribution
        cap_zeros_tensor = torch.zeros((proc_frame.shape))
        cap_eye_tensor = torch.eye(proc_frame.shape[-1])

        cap_noise = D.MultivariateNormal(cap_zeros_tensor, cap_eye_tensor)

        # generate noise from image for energy calc
        cap_gen_noise = cap_noise.sample()

        # get energy from just the frame
        # load energy based model with frame tensor width as dim
        ebm = EBM(dim=proc_frame.shape[2])
        frame_energy, frame_energy_acc, frame_tensor_energy = ebm.nce_loss(
            cap_noise, 
            proc_frame, 
            cap_gen_noise
        )
        
        logging.info(f"frame_energy: {frame_energy}")
        frame_energy_list.append((frame_cnt, float(frame_energy)))
        fe_func_img = draw_function_image(
            frame_energy_list,
            "Frame Energy by Frame",
            x_title="Frame",
            y_title="Energy" 
        )

        logging.info(f"frame_energy_acc: {frame_energy_acc}")
        logging.info(f"frame_tensor_energy.shape {frame_tensor_energy.shape}")

        # resize 
        frame_tensor_energy = cv2.resize(
            frame_tensor_energy.squeeze().cpu().detach().numpy(),
            ((proc_frame.shape[2], proc_frame.shape[3]))
        )

        logging.info(f"resize frame_tensor_energy.shape {frame_tensor_energy.shape}")

        # --------- AlexNet feature energy ---------- #

        # get alexnet features from image
        an_features = anf(proc_frame)

        logging.info(f"an_features.shape {an_features.shape}")

        # load gaussian distribution
        feature_zeros_tensor = torch.zeros((an_features.shape))
        feature_eye_tensor = torch.eye(an_features.shape[-1])

        feature_noise = D.MultivariateNormal(feature_zeros_tensor, feature_eye_tensor)

        # generate noise from image for energy calc
        feature_gen_noise = feature_noise.sample()

        # get energy from features
        ebm = EBM(dim=an_features.shape[2])
        feature_energy, feature_energy_acc, feature_tensor_energy = ebm.nce_loss(
            feature_noise,
            an_features,
            feature_gen_noise
        )

        logging.info(f"feature_energy_acc: {feature_energy_acc}")
        logging.info(f"features_energy: {feature_energy}\n")

        feature_energy_list.append((frame_cnt, float(feature_energy)))
        afe_img = draw_function_image(
            feature_energy_list,
            "AlexNet Feature Energy by Frame",
            x_title="Frame",
            y_title="Energy" 
        )
        
        logging.info(f"feature_tensor_energy.shape {feature_tensor_energy.shape}")

        # resize features energy tensor
        feature_tensor_energy = cv2.resize(
            feature_tensor_energy.squeeze().cpu().detach().numpy(),
            (proc_frame.shape[2], proc_frame.shape[3])
        )

        logging.info(f"resized feature_tensor_energy.shape {feature_tensor_energy.shape}")

        
        # generate 3d graph images via matplotlib
        ftd_img = visualize_tensor_3d(
            cap_array, 
            'Image Frame Tensor'
        )

        fe_img = visualize_tensor_3d(
            # frame_tensor_energy.detach().numpy(),
            frame_tensor_energy,
            'Energy Tensor of Image Frame'
        )

        fte_img = visualize_tensor_3d(
            feature_tensor_energy,
            'Energy Tensor of AlexNet Features via EBM'
        )
        

        # show caption if any in log
        if cap_frame[2] != "":
            logger.info(f"\ncaption: {cap_frame[2]}\n")
        
        # display graphics
        cv2.imshow("Frame", cap_array)
        cv2.setWindowProperty("Frame", 1, cv2.WINDOW_NORMAL) # for resizing
        cv2.resizeWindow("Frame", 224, 224)
        cv2.imshow("FTD", ftd_img)
        cv2.setWindowProperty("FTD", 1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FTD", 512, 512)
        cv2.imshow("FE", fe_img)
        cv2.setWindowProperty("FE", 1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FE", 512, 512)
        cv2.imshow("FEF", fe_func_img)
        cv2.setWindowProperty("FEF", 1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FEF", 512, 512)
        cv2.imshow("FTE", fte_img)
        cv2.setWindowProperty("FTE", 1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FTE", 512, 512)
        cv2.imshow("AFE", afe_img)
        cv2.setWindowProperty("AFE", 1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AFE", 512, 512)

        frame_cnt += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()