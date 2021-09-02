import cv2
import matplotlib.pyplot as plt
import numpy as np
from save_file import save_file


class Image_slice:
    def __init__(self, image_slice, x_position, y_position, x_span, y_span):
        self.image = image_slice
        self.x = x_position
        self.y = y_position
        self.x_span = x_span
        self.y_span = y_span
        self.left_houghline_pixel = 0
        self.right_houghline_pixel = 0
        self.width = 0

    def print_basic_information(self):
        self.__print_sentence("size", (self.x_span, self.y_span))
        self.__print_sentence("position", (self.x, self.y))

    def binarization(self, method, bin_method, blocksize, costant):
        self.binarized_image = cv2.adaptiveThreshold(self.image, 255, method, bin_method, blocksize, costant)

    def detect_houghlines_vertical(self, threshold):
        self.point = []
        for x in range(self.y_span):
            lines = cv2.HoughLines(self.binarized_image[:, x], 1, np.pi / 180, threshold)
            if lines is not None:
                self.point.append(x)
        self.__print_sentence("houghlines_x_pixel", self.point)

    def parse_houghlines(self, scale, test=False):
        difference = []
        for index, item in enumerate(self.point):
            if index == 0:
                continue
            temp = item - self.point[index - 1]
            if temp >= 5:
                self.left_houghline_pixel = self.point[index - 1]
                self.right_houghline_pixel = item
                difference.append(temp)

        if test:
            self.__print_sentence("pixel width of pitch", difference)
            self.__print_sentence("width of pitch", difference[0] * scale)
        else:
            self.width = difference[0] * scale
            self.__print_sentence("width of pitch", self.width)
            self.__print_sentence("pixel of left and right hough line pixel",
                                  (self.left_houghline_pixel, self.right_houghline_pixel))

    def paint_line(self):
        """
        draw recognized hough lines to the image
        :param x_value:two element array, consisted of left and right line of the pitch
        :type x_value:array
        :return: none
        :rtype:none
        """
        self.color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        left = self.left_houghline_pixel
        right = self.right_houghline_pixel
        cv2.line(self.color_image, (left, 0), (left, self.x_span), (0, 255, 0), 5)
        cv2.line(self.color_image, (right, 0), (right, self.x_span), (0, 255, 0), 5)

    def paint_width(self):
        """
        draw width number to the picture
        :param width: pitch width recognized form the hough lines
        :type width: float
        :return: none
        :rtype: none
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        # first decimal decrease, the text will move left
        # second decimal decrease, the text will move up
        bottomLeftCornerOfText = (round(0.01 * self.y_span), round(0.3 * self.x_span))
        fontScale = 1.7
        fontColor = (255, 0, 0)
        lineType = 2

        cv2.putText(self.color_image, str("{:.3f}".format(self.width)),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor, 4, lineType)

    def __print_sentence(self, name, value):
        sentence = f"image on ({self.x}, {self.y}) ---- Properties: {name}, Values: {value}."
        print(sentence)


def pre_slice(image: object, is_draw: bool = True, bin_method=0, bin_region=0, bin_counts=0,
              hough_threshold=0) -> object:
    """
    ensure the slice X axis values
    :param image: image prepared to slice
    :type image: opencv. image .object
    :return: slice X-array position
    :rtype:array
    """

    thresh2 = cv2.adaptiveThreshold(image, 255, bin_method, cv2.THRESH_BINARY, bin_region, bin_counts)
    axes2_span = thresh2.shape[1]

    # detect lines
    axes2_position = []
    for axes2 in range(axes2_span):
        temp_image = thresh2[:, axes2]
        lines = cv2.HoughLines(temp_image, 1, np.pi / 180, hough_threshold)
        if lines is not None:
            axes2_position.append(axes2)

    print(f"axes2_position: {axes2_position}")

    # detect pitch
    pitch_position = []
    pitch_width = []
    for index, item in enumerate(axes2_position):
        if index == 0:
            continue
        temp = item - axes2_position[index - 1]
        if temp >= 10:
            pitch_position.append(axes2_position[index - 1])
            pitch_position.append(item)
            pitch_width.append(temp)
    print(f"pitch_position: {pitch_position}")
    print(f"pitch_width: {pitch_width}")

    # parse the separate line position
    slice_line_position = []
    for index, item in enumerate(pitch_position):
        # jump the first line
        if index == 0:
            continue
        # if the first line is the left line, jump the odd index line.
        # make sure the image start with a diamond part, not in a pitch
        if index % 2 == 1:
            continue

        temp = round((item + pitch_position[index - 1]) / 2)
        slice_line_position.append(temp)
    print(f"slice_line_position: {slice_line_position}")

    if is_draw:
        # draw lines on the image
        plt.figure()
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for item in slice_line_position:
            cv2.line(color_image, (item, 0), (item, color_image.shape[0]), (255, 0, 0), 10)
        plt.imshow(color_image)
        plt.show()

    return slice_line_position


def get_width(x_array, scale):
    """
    depending on the pixel difference and the scale bar of SEM, calculate the actual width.
    :param x_array: pixel array
    :type x_array: list
    :param scale: width of per pixel, for example,  5 um / 134 (pixel)
    :type scale: float
    :return: width array
    :rtype: numpy array
    """
    difference = []
    for index, item in enumerate(x_array):
        if index == 0:
            continue
        temp = item - x_array[index - 1]
        if temp != 1:
            difference.append(temp)
    return np.array(difference) * scale


def separate_image(image, x_item=0, y_item=0, x_slice_array=[], y_slice_array=[]):
    # if x_item and y_item:
    #     x_shape = image.shape[0]
    #     y_shape = image.shape[1]
    #
    #     x_span = int(x_shape / x_item)
    #
    #     image_list = np.zeros((x_item, y_item), dtype=object)
    #
    #     y_span = int(y_shape / y_item)
    #
    #     for x in range(x_item):
    #         for y in range(y_item):
    #             x_start = x * x_span
    #             y_start = y * y_span
    #
    #             temp_image = Image_slice(image[x_start: (x_start + x_span), y_start: (y_start + y_span)], x, y, x_span,
    #                                      y_span)
    #             image_list[x, y] = temp_image
    #             del temp_image

    print(image.shape)

    x_item = len(x_slice_array)
    y_item = len(y_slice_array)
    image_list = np.zeros((x_item, y_item), dtype=object)

    for index_x, item_x in enumerate(x_slice_array):

        if index_x == 0:
            x_start = 0
            x_end = item_x
        else:
            x_start = x_slice_array[index_x - 1]
            x_end = item_x

        for index_y, item_y in enumerate(y_slice_array):
            if index_y == 0:
                y_start = 0
                y_end = item_y
            else:
                y_start = y_slice_array[index_y - 1]
                y_end = item_y

            temp_image = Image_slice(image[x_start: x_end,
                                     y_start:y_end], index_x, index_y,
                                     x_end - x_start,
                                     y_end - y_start)

            image_list[index_x, index_y] = temp_image

            del temp_image

    return image_list


def image_deal(image_list, method=0, bin_method=0, bin_region=0, bin_counts=0, hough_line_threshold=0, scale=10 / 640):
    """
    The image deal method all list here
    :param image_list:  sliced image array
    :type image_list: Image_slice array
    :return: image after dealing
    :rtype: Image_slice
    """
    items_x = image_list.shape[0]
    items_y = image_list.shape[1]
    for x in range(items_x):
        for y in range(items_y):
            sliced_image = image_list[x, y]
            # for single picture, do some thing like binarization, houghline detection and so no.
            sliced_image.binarization(method, bin_method, bin_region, bin_counts)
            # temp_image.print_basic_information()

            # threshold small, numbers of line will increase, the width will be decrease.

            threshold = hough_line_threshold
            while True:
                flag = 0
                try:

                    # try the (threshold, threshold + 20) hough line detection
                    width_array = []
                    for i in range(20):
                        left_houghline_pixel = 0
                        right_houghline_pixel = 0
                        width = 0
                        sliced_image.point = []

                        sliced_image.detect_houghlines_vertical(threshold + 1 * i)

                        if sliced_image.point == []:
                            continue
                        for index, item in enumerate(sliced_image.point):
                            if index == 0:
                                continue
                            temp = item - sliced_image.point[index - 1]

                            if temp >= 45:
                                left_houghline_pixel = sliced_image.point[index - 1]
                                right_houghline_pixel = item
                                width = (right_houghline_pixel - left_houghline_pixel ) * scale
                                break

                        width_array.append([width, left_houghline_pixel, right_houghline_pixel])

                    # find max width in the range of (50, 50 + 20) threshold
                    sliced_image.width = 0
                    print(width_array)
                    for item in width_array:
                        if item[0] > sliced_image.width:
                            sliced_image.width = item[0]
                            sliced_image.left_houghline_pixel = item[1]
                            sliced_image.right_houghline_pixel = item[2]


                    print(f"({sliced_image.x, sliced_image.y}) left pixel: {sliced_image.left_houghline_pixel}")
                    print(
                        f"({sliced_image.x, sliced_image.y}) right pixel: {sliced_image.right_houghline_pixel}")
                    print(f"({sliced_image.x, sliced_image.y}) width: {sliced_image.width}")

                    if sliced_image.width >= (50 * scale) and sliced_image.width <= (90 * scale):
                        flag = 1
                        break
                except:
                    print("no value was found, repeat input")
                finally:
                    if flag != 1:
                        sliced_image.point = []
                        sliced_image.width = None
                        sliced_image.houghlines_x_pixel = None
                        sliced_image.left_houghline_pixel = None
                        sliced_image.right_houghline_pixel = None
                        threshold = int(input("输入标准，空格间隔："))
                        continue
                    else:
                        break

            sliced_image.paint_line()
            sliced_image.paint_width()


def plot_figure(image_list):
    items_x = image_list.shape[0]
    items_y = image_list.shape[1]
    i = 1
    plt.figure()
    for x in range(items_x):
        for y in range(items_y):
            plt.subplot(items_x, items_y, i)
            i += 1
            plt.imshow(image_list[x, y].color_image, cmap="gray")
            plt.xticks([]), plt.yticks([])
    plt.show()


# def concat_image(image_slice_array, separate_size):
#     x_item = image_slice_array[-1].x + 1
#     y_item = image_slice_array[-1].y + 1
#
#     x_span = image_slice_array[-1].x_span
#     y_span = image_slice_array[-1].y_span
#
#     canvas_size_x = image_slice_array[0, 0].size[0] * x_item + (x_item - 1) + separate_size
#     canvas_size_y = image_slice_array[0, 0].size[1] * y_item + (y_item - 1) + separate_size
#     canvas = np.zeros((canvas_size_x, canvas_size_y))
#
#     # define the default background gray degree value
#     canvas[:, :] = 75
#
#     for x in range(x_item):
#         for y in range(y_item):
#             canvas[x:x_span, y:y_span] = image_slice_array[]

# def slice(image):
#     slice_array = [65, 145, 225, 305, 385, 465, 540, 620, 700, 770, 850, 930, 1008]
#
#     width, image_list = separate_image(image[0:1000, 0:1008], 20, slice_array=slice_array)
#
#     # i is for subplot
#     i = 1
#     for x in range(image_list.shape[0]):
#         for y in range(image_list.shape[1]):
#             plt.subplot(10, len(slice_array) - 1, i)
#             i += 1
#             plt.imshow(image_list[x, y].color_image)
#             plt.xticks([]), plt.yticks([])
#         # for item in slice_array:
#         #     plt.axvline(x=item)
#     plt.show()
#     plt.figure()
#     n, bins, patches = plt.hist(width)
#     mu = np.mean(width)
#     sigma = np.std(width)
#     gussian_y = norm.pdf(bins, mu, sigma)
#     plt.plot(bins, gussian_y)
#     plt.show()

def size_control(sliced_image, first_threshold):
    sliced_image.binarization(cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -15)

    # threshold small, numbers of line will increase, the width will be decrease.
    sliced_image.detect_houghlines_vertical(first_threshold)

    for index, item in enumerate(sliced_image.point):
        if index == 0:
            continue
        else:
            if item - sliced_image.point[index - 1] > 20:
                sliced_image.left_houghline_pixel = sliced_image.point[index - 1]

    if not sliced_image.left_houghline_pixel:
        sliced_image.left_houghline_pixel = sliced_image.point[-1]

    print(f"({sliced_image.x, sliced_image.y}) left pixel: {sliced_image.left_houghline_pixel}")

    sliced_image.binarization(cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, -15)
    sliced_image.detect_houghlines_vertical(second_threshold)

    sliced_image.right_houghline_pixel = sliced_image.point[-1]

    print(f"({sliced_image.x, sliced_image.y}) right pixel: {sliced_image.right_houghline_pixel}")

    sliced_image.width = (sliced_image.right_houghline_pixel - sliced_image.left_houghline_pixel) * 10 / 640
    print(f"width(64-1.0, 83-1.3): {sliced_image.width}")


if __name__ == '__main__':
    # image = cv2.imread(r"./assets/SEM_pr_134_5um.jpg", flags=cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(r"./assets/dia_5k_10um_640.tif", flags=cv2.IMREAD_GRAYSCALE)
    print("the primary image shape:" + str(image.shape))

    # ===========================================================
    # single picture for test argument.
    # ===========================================================

    # . 1. first choose the proper region of the new figure;
    # . 2. choose a slice include all the rows, figure[:, x:y], the region {x,Y} should include a periodicity.
    # . 3. Adjusting the threshold of hough lines.
    # . 4. Apply the arguments to pre_slice function to one step further to slice the image.
    # . 5. for single slice image. (figure[x1:y1, x2:y2]), adjustment the argument.
    #
    new_image = image[0:3313, 0:-1]
    #
    # test_image = Image_slice(new_image, 0, 0, new_image.shape[0], new_image.shape[1])
    # test_image.print_basic_information()
    #
    # test_image.binarization(cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -15)
    # test_image.detect_houghlines_vertical(40)
    # # test_image.parse_houghlines(5 / 134, test=True)
    #
    # test_image.binarization(cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, -15)
    # test_image.detect_houghlines_vertical(130)
    # thresh2 = test_image.binarized_image

    # plt.subplot(1, 3, 1)
    # plt.imshow(new_image, cmap="gray")
    # plt.subplot(1, 3, 2)
    # plt.imshow(test_image.binarized_image, cmap="gray")
    # plt.subplot(1, 3, 3)
    # plt.imshow(thresh2, cmap="gray")
    # plt.show()

    # ===========================================================
    # multi picture for use.
    # ===========================================================

    # 1. choose the proper image part
    # new_image = image[0:3313, 0:-1]
    # 2. pre-slice image, using hough lines
    # slice_line_position = pre_slice(new_image, is_draw=True, bin_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, bin_region=21,
    # bin_counts=-15, hough_threshold=900)

    # pr_5K_slice_array
    # x_slice_line_position = [186, 373, 549, 738, 927, 1109, 1290, 1474, 1660, 1853, 2028, 2224, 2414, 2589, 2772, 2954,
    #                          3137]
    # y_slice_line_position = [186, 373, 549, 738, 927, 1109, 1290, 1474, 1660, 1853, 2028, 2224, 2414, 2589, 2772, 2954,
    #                          3137, 3333, 3515, 3697, 3887, 4076, 4258, 4444, 4630, 4819]
    # SEM_Pr_200_10um_77pixel.jpg
    # x_slice_line_position = [22, 44, 66, 88, 111, 133, 155, 178, 200, 222, 244, 267, 289, 312, 334, 356, 378, 400, 422, 445, 467, 490, 512, 534, 556, 578, 600, 623, 645, 667, 690, 712, 734, 756, 778, 801, 823, 845, 867, 890]
    # y_slice_line_position = [22, 44, 66, 88, 111, 133, 155, 178, 200, 222, 244, 267, 289, 312, 334, 356, 378, 400, 422, 445, 467, 490, 512, 534, 556, 578, 600, 623, 645, 667, 690, 712, 734, 756, 778, 801, 823, 845, 867, 890, 912, 934, 956, 980]
    #
    # diamond_5K_array
    x_slice_line_position = [158, 344, 529, 715, 899, 1085, 1271, 1458, 1644, 1829, 2015, 2199, 2384, 2571, 2757, 2942,
                             3128, 3313]
    y_slice_line_position = [158, 344, 529, 715, 899, 1085, 1271, 1458, 1644, 1829, 2015, 2199, 2384, 2571, 2757, 2942,
                             3128, 3313, 3498, 3684, 3869, 4054, 4240, 4425, 4612, 4797, 4983]

    # 3. separate_image and deal
    image_list = separate_image(new_image, x_slice_array=x_slice_line_position, y_slice_array=y_slice_line_position)

    image_list = image_list

    image_deal(image_list, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, bin_method=cv2.THRESH_BINARY, bin_region=41, bin_counts= -15, hough_line_threshold=55, scale= 10/640)

    # width = []
    #
    # items_x = image_list.shape[0]
    # items_y = image_list.shape[1]
    #
    # for x in range(items_x):
    #     for y in range(items_y):
    #         sliced_image = image_list[x, y]
    #         # for single picture, do some thing like binarization, houghline detection and so no.
    #
    #         first = 40
    #         # second = 140
    #         while True:
    #             flag = 0
    #             try:
    #                 size_control(sliced_image, int(first))
    #             except:
    #                 print("no value was found, repeat input")
    #                 first, second = input("输入 2 个标准，空格间隔：").split(" ")
    #             else:
    #                 order = input("确认结果？")
    #                 if order == "1":
    #                     flag = 1
    #                 else:
    #                     first, second = input("输入 2 个标准，空格间隔：").split(" ")
    #             finally:
    #                 if not flag:
    #                     sliced_image.point = []
    #                     sliced_image.width = None
    #                     sliced_image.houghlines_x_pixel = None
    #                     sliced_image.left_houghline_pixel = None
    #                     sliced_image.right_houghline_pixel = None
    #                     continue
    #                 else:
    #                     break
    #
    #         sliced_image.paint_line()
    #
    #         sliced_image.paint_width()
    #         width.append(sliced_image.width)
    #
    plot_figure(image_list)
    save_file(image_list)
    # =============================================================================

    # thresh2 = cv2.adaptiveThreshold(image[0:1000, 0:1000], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -5)
    # ret3, thresh3 = cv2.threshold(image[0:1000, 0:1000], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(thresh2)
    #
    # # vertical_image = thresh2[:, 45]
    # # plt.plot(vertical_image)
    # point = []
    # for x in range(1000):
    #     lines = cv2.HoughLines(thresh2[:, x], 1, np.pi / 180, 350)
    #     if lines is not None:
    #         point.append(x)
    # print(point)
    #
    # final = get_width(point, 5 / 134)
    # print(final)

    # contours_2, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    # contours_3, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    # cv2.drawContours(thresh2, contours_2, -1, (211,82,48), 10)
    # cv2.drawContours(thresh3, contours_3, -1, (211,82,48), 10)

    # canny
    # edges = cv2.Canny(thresh2, 100, 200, apertureSize=7)

    # hough lines, find shapes depending on the iteration
    # lines = cv2.HoughLines(image_vertical, 1, np.pi / 180, 420)
    # lines = cv2.HoughLinesP(image_vertical, 1, np.pi / 180, 400, 1000, 4)

    # (27259, 1, 2)
    # print(lines)

    # def draw_lines(img, lines):
    #     for line in lines:
    #         coords = line[0]
    #         cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)

    # vertical_lines = lines[lines[:, 0, 1] == 0 /180 * np.pi]
    # print(vertical_lines.shape)
    # draw_lines(processed_img, vertical_lines)

    # vertical_lines[:,0,0] rho, [:, 0, 1] theta
    # a = np.cos(vertical_lines[:, 0, 1])
    # b = np.sin(vertical_lines[:, 0, 1])
    # x0 = a * vertical_lines[:, 0, 0]
    # y0 = b * vertical_lines[:, 0, 0]
    # x1 = int(x0 + 1000 * )

    # for item in lines[lines[:, 0, 1] == 0]:
    #     for rho, theta in item:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #
    #         cv2.line(thresh2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    #     break
    #
    # images = [image, thresh2, thresh3, edges]
    # for index, item in enumerate(images):
    #     plt.subplot(2, 4, index + 1)
    #     plt.imshow(item, cmap="gray")
    #     # plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.subplot(2, 1, 1)
    # plt.imshow(cv2.cvtColor(image_vertical, cv2.COLOR_BGR2RGB))
    # plt.subplot(2, 1, 2)
    # plt.imshow(cv2.cvtColor(thresh3, cv2.COLOR_BGR2RGB))
    # plt.show()

# canny edge detection
# edges = cv2.Canny(image, 200, 30)
#
# plt.subplot(121),plt.imshow(image,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# binarization
# ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -50)

# print image
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

# contours
# contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#
# contours = np.array(contours)
# print(contours.shape)
