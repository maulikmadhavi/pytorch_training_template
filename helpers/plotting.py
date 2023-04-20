"""
This script stores all the helper functions related to plotting.

"""
import torch
import numpy as np
import cv2
import torchvision
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


# def get_cam(model, dataloader):

#     # im = cv2.imread('/home/maulik/practice/pytorch_templates/cat-2083492_960_720.jpg')
#     # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     # # use the ImageNet transformation
#     # transform = transforms.Compose([transforms.ToPILImage(),
#     #                                 transforms.Resize((224, 224)),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     # img = transform(im).unsqueeze(0)
#     # pred = model(image)

#     # get the gradient of the output with respect to the parameters of the model
#     breakpoint()
#     # Randomly select a sample from the dataloader
#     model.cpu()
#     image, label = dataloader.dataset[torch.randint(0, len(dataloader.dataset), (1,))]
#     image = image.unsqueeze(0)  # NCHW
#     prediction = model(image)
#     prediction[:, label].backward()  # NCHW

#     # pull the gradients out of the model
#     gradients = model.get_activations_gradient()

#     # pool the gradients across the channels
#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

#     # get the activations of the last convolutional layer
#     activations = model.get_activations(image).detach()

#     # weight the channels by corresponding gradients
#     for i in range(512):
#         activations[:, i, :, :] *= pooled_gradients[i]

#     # average the channels of the activations
#     heatmap = torch.mean(activations, dim=1).squeeze()

#     # relu on top of the heatmap
#     # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
#     heatmap = np.maximum(heatmap, 0)

#     # normalize the heatmap
#     heatmap /= torch.max(heatmap)

#     # draw the heatmap
#     # plt.matshow(heatmap.squeeze())

#     heatmap = cv2.resize(heatmap.numpy(), (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     # img12 = inv_normalize(img[0])
#     img12 = inv_normalize(image[0]).transpose(0, 2).transpose(0, 1)
#     superimposed_img = heatmap * 0.4 + np.uint8(255 * img12)
#     # cv2.imwrite("./map281.jpg", superimposed_img)
#     # mn = np.min(superimposed_img)
#     # mx = np.max(superimposed_img)
#     # superimposed_img1 = (superimposed_img - mn) / (mx - mn)
#     return torchvision.transforms.ToTensor()(superimposed_img)  # CHW


def get_cam(model, dataloader):
    model.cpu()
    image, label = dataloader.dataset[torch.randint(0, len(dataloader.dataset), (1,))]
    targets = [ClassifierOutputTarget(label)]
    target_layers = [model.model.layer4[-1]]
    input_tensor = image.unsqueeze(0)
    img = inv_normalize(image).transpose(0, 2).transpose(0, 1).numpy()
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    # cam = np.uint8(255 * grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    # images = np.hstack((np.uint8(255 * img), cam, cam_image))
    return torchvision.transforms.ToTensor()(
        cam_image
    ), torchvision.transforms.ToTensor()(img)
