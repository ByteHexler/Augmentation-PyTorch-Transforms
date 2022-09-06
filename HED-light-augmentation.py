import myTransforms as transforms

# data augmentation like HED-light in:
# Tellez, David, et al.
# "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology."
# Medical image analysis 58 (2019): 101544.

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomChoiceRotation(),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomElastic(alpha=[140,210], sigma=[16,19]),
        transforms.RandomResize((0.9, 1.1)),
        transforms.RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.35),
            transforms.ColorJitter(contrast=0.5),
            transforms.HEDJitter(0.05, mode='HE'),
        ]),
        transforms.RandomGaussBlur((0,0.1)),
        transforms.ToTensor(),
        transforms.Normalize(means_and_stds['channel_means'], means_and_stds['channel_stds']),
        transforms.RandomGaussNoise((0,0.1))
    ]),
    'val': transforms.Compose([             # transforms without random components
        transforms.ToTensor(),
        transforms.Normalize(means_and_stds['channel_means'], means_and_stds['channel_stds'])
    ]),
}
