import SimpleITK as sitk
import numpy  as np

def PreprocessTrainingData(data_files, target_files, input_size):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    data = []
    targ = []
    for i in range(len(data_files)):
        # Load data
        reader.SetFileName(data_files[i])
        temp_data1 = reader.Execute()
        # temp_data1 = sitk.ReadImage(data_files[i])
        a = sitk.GetArrayFromImage(temp_data1)
        temp_data = sitk.GetImageFromArray(a[0,...])
        temp_data.SetSpacing(temp_data1.GetSpacing()[0:3])
        # Load Target
        reader.SetFileName(target_files[i])
        temp_targ = reader.Execute()
        # temp_targ = sitk.ReadImage(target_files[i])


        # Perform Cubic Resampling
        rs_data = sitk.GetArrayFromImage(cubic_resample(temp_data, spacing=1.0))
        rs_targ = sitk.GetArrayFromImage(cubic_resample(temp_targ, spacing=1.0))

        for i1 in range(rs_data.shape[0]):
            if rs_targ[i1,...].max() >1:
                pd_data, pd_targ = \
                    crop_images_centered_over_label(rs_data[i1,...],
                                                    rs_targ[i1,...],
                                                    input_size)
            else:
                pd_data = pad_image(rs_data[i1,...],shape=input_size,dim=2)
                pd_targ = pad_image(rs_targ[i1,...],shape=input_size,dim=2)
            a = normalize(pd_data)
            b = np.nan_to_num(a)
            data.append(b)
            targ.append((pd_targ==2).astype(np.uint8))

    return data, targ


def cubic_resample(image, spacing=2.0):
    """Resample volume to cubic voxels using sitk routines

    Args:+
        image: sitk image
        spacing:

    Returns:
        sitk image

     """
    resample = sitk.ResampleImageFilter()

    spacingOut = [spacing, spacing, spacing]
    resample.SetOutputSpacing(spacingOut)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    spacingIn = image.GetSpacing()
    shape = image.GetSize()

    newSize = [int(shape[0] * spacingIn[0] / spacingOut[0]),
               int(shape[1] * spacingIn[1] / spacingOut[1]),
               int(shape[2] * spacingIn[2] / spacingOut[2])]

    resample.SetSize(newSize)
    new = resample.Execute(image)

    return new


def pad_image(np_im, shape, dim):
    """ Takes an image and pads or crops to fit standard volume

    Args:
        np_im: numpy array
        shape: [nslices,nrows,ncols]

    Returns:
        np_padded : new numpy array

    """

    np_padded = np.zeros(shape, dtype=np_im.dtype)
    old_shape = np_im.shape
    pad_llims = np.zeros([dim], dtype=np.uint8)
    pad_ulims = np.zeros([dim], dtype=np.uint8)
    old_llims = np.zeros([dim], dtype=np.uint8)
    old_ulims = np.zeros([dim], dtype=np.uint8)

    for i in range(dim):
        if shape[i] < old_shape[i]:  # need to crop input image
            pad_llims[i] = 0
            pad_ulims[i] = shape[i]
            crop = int((old_shape[i] - shape[i]) / 2)
            old_llims[i] = crop
            old_ulims[i] = crop + shape[i]
        elif shape[i] == old_shape[i]:  # need to crop input image
            pad_llims[i] = 0
            pad_ulims[i] = shape[i]
            old_llims[i] = 0
            old_ulims[i] = shape[i]
        else:
            old_llims[i] = 0
            old_ulims[i] = old_shape[i]
            pad = int((shape[i] - old_shape[i]) / 2)
            pad_llims[i] = pad
            pad_ulims[i] = pad + old_shape[i]
    if dim==3:
        np_padded[pad_llims[0]: pad_ulims[0],
        pad_llims[1]: pad_ulims[1],
        pad_llims[2]: pad_ulims[2]] = \
            np_im[old_llims[0]: old_ulims[0],
            old_llims[1]: old_ulims[1],
            old_llims[2]: old_ulims[2]]
    else:
        np_padded[pad_llims[0]: pad_ulims[0],
        pad_llims[1]: pad_ulims[1]] = \
            np_im[old_llims[0]: old_ulims[0],
            old_llims[1]: old_ulims[1]]

    return np_padded

def normalize(img):
    upper_lim = 99.9
    top = np.percentile(img,upper_lim)
    img[np.where(img > top)] = top
    img = (img * 255./top).astype(np.uint8)
    return img

def crop_images_centered_over_label(img, label, sample_size):

    x_sample_size = sample_size[0]
    y_sample_size = sample_size[1]

    label_idx = np.argwhere(label == 2)

    (ymin, xmin), (ymax, xmax) = label_idx.min(0), label_idx.max(0) + 1

    xlength = xmax - xmin
    ylength = ymax - ymin

    # Recenter cropped images
    hx = int((x_sample_size - xlength) / 2)
    hy = int((y_sample_size - ylength) / 2)


    # Initial index for cropping
    crop_x0 = xmin - hx
    crop_y0 = ymin - hy

    # Check indicies for cropping to ensure they are within the image

    # Check to see if ending index is outside image
    if crop_x0 + x_sample_size > img.shape[0]:

        # If crop ending index is outside the image, adjust initial index based on difference
        diff_x0 = crop_x0 - (img.shape[0] - x_sample_size)

        crop_x0 = crop_x0 - diff_x0

    # Repeat for y direction
    if crop_y0 + y_sample_size > img.shape[1]:
        diff_y0 = crop_y0 - (img.shape[1] - y_sample_size)

        crop_y0 = crop_y0 - diff_y0

    # Ensure initial crop does not have a negative index based on adjustment. If so, set to zero
    if crop_x0 < 0:
        crop_x0 = 0

    if crop_y0 < 0:
        crop_y0 = 0

    crop_x1 = crop_x0 + x_sample_size
    crop_y1 = crop_y0 + y_sample_size

    cropped_img = img[crop_x0:crop_x1, crop_y0:crop_y1]
    cropped_label = label[crop_x0:crop_x1, crop_y0:crop_y1]

    return cropped_img, cropped_label