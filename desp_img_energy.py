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

import time

from captures import Captures
from alexnet_features import AlexNetFeatures
from ebm import EBM

def preprocess_image(image):
    # using imagenet mean and std
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def generate_negative_samples(image, num_samples=1):
    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    # Generate random transformations
    img_transforms = [
        # transforms.RandomRotation(degrees=(-10, 10)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ]

    # Apply random transformations to the image
    negative_samples = []
    for _ in range(num_samples):
        # Randomly select a transform
        transform_idx = np.random.randint(len(img_transforms))
        img_transform = img_transforms[transform_idx]

        # Apply the selected transform to the image tensor
        negative_sample = img_transform(image_tensor).squeeze().permute(1, 2, 0).numpy()
        negative_samples.append(negative_sample)

    return np.array(negative_samples)

def visualize_img_energy_map(
    image,
    temperature=1.0, 
    num_negative_samples=10
):
    cframe = image
    cframe = torch.from_numpy(cframe).float()

    # Generate negative samples
    # negative_samples = generate_negative_samples(image, num_samples=num_negative_samples)
    # negative_samples = torch.from_numpy(image).float()

    # generate noise samples


    # Calculate the NCE loss
    nce_loss = tnf.binary_cross_entropy_with_logits(
        cframe, 
        torch.ones_like(cframe), 
        reduction='none'
    )
    # nce_loss += tnf.binary_cross_entropy_with_logits(
    #     negative_samples, 
    #     torch.zeros_like(negative_samples), 
    #     reduction='none'
    # ).mean(dim=0)

    # Calculate the energy map
    energy_map = -temperature * nce_loss.numpy()

    # Normalize the energy map
    energy_map = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply colormap for visualization
    energy_map = cv2.applyColorMap(energy_map, cv2.COLORMAP_JET)

    # Overlay the energy map on the original image
    overlaid_image = cv2.addWeighted(image, 0.7, energy_map, 0.3, 0)

    return overlaid_image

def visualize_img_energy_map_3d(
    image,
    temperature=1.0,
    num_negative_samples=10
):
    cframe = image
    cframe = torch.from_numpy(cframe).float()

    # Generate negative samples
    # negative_samples = generate_negative_samples(image, num_samples=num_negative_samples)
    # negative_samples = torch.from_numpy(negative_samples).float()

    # Initialize the energy map
    energy_map = np.zeros((image.shape[0], image.shape[1]))
    height, width = energy_map.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the NCE loss
    nce_loss = tnf.binary_cross_entropy_with_logits(
        cframe, torch.ones_like(cframe), reduction='none'
    )
    # nce_loss += tnf.binary_cross_entropy_with_logits(
    #     negative_samples, torch.zeros_like(negative_samples), reduction='none'
    # ).mean(dim=0)

    # Reshape the nce_loss tensor to match the shape of the input image
    nce_loss = nce_loss.view(image.shape[2], image.shape[0], image.shape[1])
    nce_loss = nce_loss.permute(1, 2, 0).contiguous()
    nce_loss = nce_loss.view(image.shape[0], image.shape[1], -1)

    # Calculate the energy map
    energy_map = -temperature * nce_loss.numpy().squeeze()

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    print(f"energy map shape: {energy_map.shape}")

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


def visualize_contour_energy_map(
        image, 
        contour_probabilities, 
        temperature=1.0, 
        num_negative_samples=10
):
    # Generate negative samples
    # negative_samples = generate_negative_samples(image, num_samples=num_negative_samples)

    # Resize the contour probabilities to match the image size
    # contour_probabilities = contour_probabilities.squeeze().cpu().numpy()
    contour_probabilities = cv2.resize(contour_probabilities, (image.shape[1], image.shape[0]))

    # Convert the contour probabilities and negative samples to PyTorch tensors
    contour_probabilities = torch.from_numpy(contour_probabilities).float()

    # Reshape negative_samples to match the shape of contour_probabilities
    # negative_samples = negative_samples.reshape(-1, *contour_probabilities.shape)
    # negative_samples = torch.from_numpy(negative_samples).float()

    # Calculate the NCE loss
    nce_loss = tnf.binary_cross_entropy_with_logits(contour_probabilities, torch.ones_like(contour_probabilities), reduction='none')
    # nce_loss += tnf.binary_cross_entropy_with_logits(negative_samples, torch.zeros_like(negative_samples), reduction='none').mean(dim=0)

    # Calculate the energy map
    energy_map = -temperature * nce_loss.numpy()

    # Normalize the energy map
    energy_map = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply colormap for visualization
    energy_map = cv2.applyColorMap(energy_map, cv2.COLORMAP_JET)

    # Overlay the energy map on the original image
    overlaid_image = cv2.addWeighted(image, 0.7, energy_map, 0.3, 0)

    return overlaid_image, energy_map

def visualize_contour_energy_3d(
    image,
    energy_map
):
    """
    3D graph display of the contour energies
    """
    # Initialize the energy map
    energy_map = np.zeros((image.shape[0], image.shape[1]))
    height, width = energy_map.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

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

def main():
    # load energy based model
    ebm = EBM()
    
    # load alexnet only features
    anf = AlexNetFeatures()

    # Load frames
    caps = Captures()
    cap_frames = caps.get_all_pframes(limit=0)
    for cap_frame in cap_frames:
        # convert blob to image array
        cap_txt = cap_frame[1]
        cap_array = caps.convert_array(cap_txt)
        cv2.imshow("Original Frame", cap_array)

        print(f"cap_array: {cap_array.shape}")

        # # Preprocess the image
        # preprocessed_image = preprocess_image(cap_array)
        # preprocessed_image = preprocessed_image.unsqueeze(0)  # Add batch dimension

        # # get alexnet features from image
        # an_features = anf(preprocessed_image)

        # print(f"\nan_features.shape: {an_features.shape}")
        
        # # load gaussian distribution
        # zeros_tensor = torch.zeros((an_features.shape))
        # eye_tensor = torch.eye(an_features.shape[-1])

        # print(f"zeros_tensor.shape {zeros_tensor.shape}")
        # print(f"eye_tensor.shape {eye_tensor.shape}")

        # noise = D.MultivariateNormal(zeros_tensor, eye_tensor)

        # # generate noise from image for energy calc
        # gen_noise = noise.sample()

        # print(f"gen_noise.shape {gen_noise.shape}")

        # # get energy
        # feature_energy, _ = ebm.nce_loss(noise, an_features, gen_noise)

        # print(f"features_energy: {feature_energy}\n")

        # alex_contours = anf(preprocessed_image)
        #     print(f"alex_contours: {alex_contours.shape}")
    
        #     # Resize the contour energies to match the input image size
        #     alex_contours = alex_contours.squeeze().cpu().numpy()
        #     alex_contours = cv2.resize(
        #         alex_contours, 
        #         (cap_array.shape[1], cap_array.shape[0])
        #     )
            
        #     # Normalize the contour energies to the range [0, 255]
        #     alex_contours = cv2.normalize(
        #         alex_contours,
        #         None,
        #         0,
        #         255,
        #         cv2.NORM_MINMAX,
        #         cv2.CV_8U
        #     )
            
        #     # Apply a colormap for visualization
        #     # contour_disp = cv2.applyColorMap(alex_contours, cv2.COLORMAP_JET)
            
        #     # Overlay the contour display on the original image
        #     # contour_highlight = cv2.addWeighted(cap_array, 0.7, contour_disp, 0.3, 0)

        # # Visualize the energy map
        # # contour_overlaid_image, energy_map = visualize_contour_energy_map(
        # #     cap_array, 
        # #     alex_contours,
        # #     0.3,
        # #     1
        # # )

        # frame_overlaid_image = visualize_img_energy_map(
        #     cap_array,
        #     np_noise,
        #     0.3,
        #     1
        # )

        # # ftd_img = visualize_img_energy_map_3d(cap_array, 1000, 5)
        # td_img = visualize_contour_energy_3d(cap_array, feature_energy)

        # # Display the original image and the energy map
        # if cap_frame[2] != "":
        #     print(f"\ncaption: {cap_frame[2]}\n")
        
        
        # # cv2.imshow("3D Energy Map of Frame", ftd_img)
        # cv2.imshow("Frame Energy Map", frame_overlaid_image)
        # # cv2.imshow("Contours AlexNet", contour_highlight)
        # # cv2.imshow("Contours Energy Image", contour_overlaid_image)
        # cv2.imshow("3D Energy Map of Contours", td_img)
        
        time.sleep(1)
    
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()