"""
DSC 20 Project
Name(s): Nayana Naineni, Ananya Wasker
PID(s):  A17929841, A17857172
Sources: None
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# # --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# # --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if type(pixels)!=list:
            raise TypeError()
        if len(pixels) == 0:
            raise TypeError()

        num_rows = len(pixels)
        num_cols = len(pixels[0])

        for row in pixels:
            if type(row)!=list:
                raise TypeError()
            if len(row)!=num_cols:
                raise TypeError()
            for pixel in row:
                if type(pixel)!=list:
                    raise TypeError()
                if len(pixel)!=3:
                    raise TypeError()
                for intensity_value in pixel:
                    if type(intensity_value)!=int:
                        raise ValueError()
                    if (intensity_value>255) or (intensity_value<0):
                        raise ValueError()

        self.pixels = pixels
        self.num_rows = num_rows
        self.num_cols = num_cols

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[list(pixel) for pixel in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if (type(row)!=int) or (type(col)!=int):
            raise TypeError()

        if (0<=row<self.num_rows) and (0<=col<self.num_cols):
            return tuple(self.pixels[row][col])
        else:
            raise ValueError()

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if (type(row)!=int) or (type(col)!=int):
            raise TypeError()

        if ((row>=0) and (row<=self.num_rows-1)) and ((col>=0) and (col<=self.num_cols-1)):
            if type(new_color)!=tuple:
                raise TypeError()
            if len(new_color)!=3:
                raise TypeError()
            for intensity_value in new_color:
                if type(intensity_value)!=int:
                    raise TypeError()

            updated_pixels = []
            for i in range(3):
                if new_color[i]>=0:
                    updated_pixels.append(new_color[i])
                else:
                    updated_pixels.append(self.pixels[row][col][i])
                if new_color[i]>255:
                    raise ValueError()
            self.pixels[row][col] = updated_pixels
        else:
            raise ValueError()

# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        inverted = [[[(255 - val) for val in pixel] for pixel in row] for row in image.get_pixels()]
        return RGBImage(inverted)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        grayscale = [[[sum(pixel) // 3] * 3 for pixel in row] for row in image.get_pixels()]
        return RGBImage(grayscale)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        rotate_180 = list(map(lambda x:x[::-1], image.get_pixels()[::-1]))
        return RGBImage(rotate_180)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        single_pixel_bright = [[(sum(pixel) // 3) for pixel in row] for row in image.get_pixels()]
        total = sum(sum(row) for row in single_pixel_bright)
        avg = total // (len(image.get_pixels()[0]) * len(image.get_pixels()))
        return avg

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if type(intensity)!=int:
            raise TypeError()
        if (intensity>255) or (intensity<-255):
            raise ValueError()
        single_pixel_adjust = lambda x:[max(0, min(255, (intensity_value + intensity))) for intensity_value in x]
        adjust_brightness = [list(map(single_pixel_adjust, row)) for row in image.get_pixels()]
        return RGBImage(adjust_brightness)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        total_pixels = image.get_pixels()
        row_count = len(total_pixels)
        column_count = len(total_pixels[0])
        blur = []

        for i in range(row_count):
            blurred_row = []
            for j in range(column_count):
                neighbors = [total_pixels[x][y] for x in range(max(0, i - 1), min(row_count, i + 2)) for y in range(max(0, j - 1), min(column_count, j + 2))]
                r_avg = sum(i[0] for i in neighbors) // len(neighbors)
                g_avg = sum(i[1] for i in neighbors) // len(neighbors)
                b_avg = sum(i[2] for i in neighbors) // len(neighbors)
                blurred_row.append([r_avg, g_avg, b_avg])
            blur.append(blurred_row)

        return RGBImage(blur)

# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        super().__init__()
        self.cost = 0
        self.num_of_coupons = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.num_of_coupons>0:
            self.num_of_coupons -= 1
        else:
            self.cost += 5
        return super().negate(image)


    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        if self.num_of_coupons>0:
            self.num_of_coupons -= 1
        else:
            self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        if self.num_of_coupons>0:
            self.num_of_coupons -= 1
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        if self.num_of_coupons>0:
            self.num_of_coupons -= 1
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        if self.num_of_coupons>0:
            self.num_of_coupons -= 1
        else:
            self.cost += 5
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if type(amount)!=int:
            raise TypeError()
        if amount<=0:
            raise ValueError()
        self.num_of_coupons+=amount

# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        if (not isinstance(chroma_image, RGBImage)) or (not isinstance(background_image, RGBImage)):
            raise TypeError()
        if chroma_image.size()!=background_image.size():
            raise ValueError()

        combined_images = []
        for i in range(chroma_image.size()[0]):
            row = []
            for j in range(chroma_image.size()[1]):
                diff = sum(abs(chroma_image.get_pixels()[i][j][k] - color[k]) for k in range(3))
                if diff < 50:
                    row.append(background_image.get_pixels()[i][j])
                else:
                    row.append(chroma_image.get_pixels()[i][j])
            combined_images.append(row)
        
        return RGBImage(combined_images)

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if type(sticker_image) != RGBImage or type(background_image)!= RGBImage:
            raise TypeError()
        if background_image.size()[0] < sticker_image.size()[0]  or background_image.size()[1] < sticker_image.size()[1]:
            raise ValueError()
        if type(x_pos) != int or type(y_pos) != int:
            raise TypeError()
        if x_pos < 0 or y_pos < 0:
            raise ValueError()
        if ((x_pos + sticker_image.size()[0]) > background_image.size()[0]) or ((y_pos + sticker_image.size()[1]) > background_image.size()[1]):
            raise ValueError()

        final_with_sticker = background_image.get_pixels()
        sticker = sticker_image.get_pixels()

        for i in range(sticker_image.size()[0]):
            for j in range(sticker_image.size()[1]):
                final_with_sticker[y_pos + i][x_pos + j] = sticker[i][j]

        return RGBImage(final_with_sticker)

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        total_pixels = image.get_pixels()
        row_count = image.size()[0]
        column_count = image.size()[1]

        calculated_pixels = [[sum(pixel) // 3 for pixel in row] for row in total_pixels]

        kernel = [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]

        highlighted = []
        for i in range(row_count):
            new_row = []
            for j in range(column_count):
                masked_value = 0
                for x in range(3):
                    for y in range(3):
                        if 0 <= i - 1 + x < row_count and 0 <= j - 1 + y < column_count:
                            masked_value += calculated_pixels[i - 1 + x][j - 1 + y] * kernel[x][y]
                masked_value = max(0, min(255, masked_value))
                new_row.append([masked_value] * 3)
            highlighted.append(new_row)

        return RGBImage(highlighted)

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors
        self.data = None

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if image1.size() != image2.size():
            raise ValueError()

        pixels1 = image1.get_pixels()
        pixels2 = image2.get_pixels()

        squared_diff_sum = sum(sum((p1[i] - p2[i]) ** 2 for i in range(3)) for row1, row2 in zip(pixels1, pixels2) for p1, p2 in zip(row1, row2))
        return (squared_diff_sum) ** (1/2)

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        max_count_dict = {}
        for i in candidates:
            if i in max_count_dict:
                max_count_dict[i]+=1
            else:
                max_count_dict[i]=1

        max_value = max(list(max_count_dict.values()))
        for key, value in max_count_dict.items():
            if value==max_value:
                return key

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if self.data is None:
            raise ValueError()
        
        distances = [(self.distance(image, k_image), label) for k_image, label in self.data]
        distances.sort(key=lambda x: x[0])
        near_labels = [i[1] for i in distances[:self.k_neighbors]]
        
        return self.vote(near_labels)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label