import numpy as np


def save_file(image_list):
    items_x = image_list.shape[0]
    items_y = image_list.shape[1]
    l_pixel = np.zeros((items_x, items_y))
    r_pixel = np.zeros((items_x, items_y))
    t_width = np.zeros((items_x, items_y))

    for x in range(items_x):
        for y in range(items_y):
            l_pixel[x, y] = image_list[x, y].left_houghline_pixel
            r_pixel[x, y] = image_list[x, y].right_houghline_pixel
            t_width[x, y] = image_list[x, y].width
    #
    np.savetxt('Results/DIA_5k/L-pixel_DIA_5K-0-3333-0-5017.csv', l_pixel, delimiter=',')
    np.savetxt('Results/DIA_5k//R-pixel_DIA_5K-0-3333-0-5017.csv', r_pixel, delimiter=',')
    np.savetxt('Results/DIA_5k//Width_DIA_5K-0-3333-0-5017.csv', t_width, delimiter=',')
    np.savetxt("Results/DIA_5k/Single_Width_DIA_5K-0-3333-0-5017.csv", t_width.ravel(), delimiter=',')
