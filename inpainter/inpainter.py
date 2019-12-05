import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from skimage.color import rgb2grey, rgb2lab
from skimage.filters import laplace
from sklearn.linear_model import Lasso
from scipy.ndimage.filters import convolve


class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.step_size = patch_size // 4

        self.plot_progress = plot_progress

        # Non initialized attributes
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.priority = None
        self.dictionary_r = []
        self.dictionary_g = []
        self.dictionary_b = []
        self.r_Lasso = Lasso(max_iter=1e6, tol=0.001)
        self.g_Lasso = Lasso(max_iter=1e6, tol=0.001)
        self.b_Lasso = Lasso(max_iter=1e6, tol=0.001)
        self.Lasso = Lasso(max_iter=1e6, tol=0.001)

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            find_start_time = time.time()
            generated_image = self._generate_by_dict(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time()-find_start_time))

            self._update_image(target_pixel, generated_image)

            keep_going = not self._finished()

        self._plot_image()
        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)  # TODO: check if this is necessary

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]

        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([height, width])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        
        for i in range(0, self.working_image.shape[0]-self.patch_size, self.step_size):
            for j in range(0, self.working_image.shape[1]-self.patch_size, self.step_size):
#                 self.dictionary_r.append(
#                         np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 0]))
#                 self.dictionary_g.append(
#                     np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 1]))
#                 self.dictionary_b.append(
#                     np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 2]))
                temp_mask = self.working_mask[i:i +
                                              self.patch_size, j:j+self.patch_size]
                if not np.any(temp_mask):
                    self.dictionary_r.append(
                        np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 0]))
                    self.dictionary_g.append(
                        np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 1]))
                    self.dictionary_b.append(
                        np.copy(self.working_image[i:i+self.patch_size, j:j+self.patch_size, 2]))
        
        self.dictionary_r = np.stack(self.dictionary_r, axis=0)
        self.dictionary_g = np.stack(self.dictionary_g, axis=0)
        self.dictionary_b = np.stack(self.dictionary_b, axis=0)
#         print("before reshape concat dictionary shape:{}".format(self.dictionary_r.shape))   
        
        self.dictionary_r = np.reshape(
            self.dictionary_r, [self.dictionary_r.shape[0], -1])
        self.dictionary_g = np.reshape(
            self.dictionary_g, [self.dictionary_g.shape[0], -1])
        self.dictionary_b = np.reshape(
            self.dictionary_b, [self.dictionary_b.shape[0], -1])
        
#         self.dictionary = np.concatenate((self.dictionary_r, self.dictionary_g, self.dictionary_b), axis=0)
        

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        # This function the center of this patch
        # Consider refactor
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal*gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2grey(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _generate_by_dict(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        
        temp_h = target_patch[0][1] - target_patch[0][0] + 1
        temp_w = target_patch[1][1] - target_patch[1][0] + 1
        
        raw_y_r = self.working_image[target_patch[0][0]:target_patch[0][1]+1,
                                      target_patch[1][0]:target_patch[1][1]+1, 0]
        raw_y_g = self.working_image[target_patch[0][0]:target_patch[0][1]+1,
                                      target_patch[1][0]:target_patch[1][1]+1, 1]
        raw_y_b = self.working_image[target_patch[0][0]:target_patch[0][1]+1,
                                      target_patch[1][0]:target_patch[1][1]+1, 2]
        
        raw_y_r = raw_y_r.reshape(-1)
        raw_y_g = raw_y_g.reshape(-1)
        raw_y_b = raw_y_b.reshape(-1)
        
        
        dictionary_tr = self.dictionary_r.reshape((-1,self.patch_size, self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w))
        dictionary_tg = self.dictionary_g.reshape((-1,self.patch_size, self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w))
        dictionary_tb = self.dictionary_b.reshape((-1,self.patch_size, self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w))
        
        temp_mask = self.working_mask[target_patch[0][0]:target_patch[0][1]+1,
                                      target_patch[1][0]:target_patch[1][1]+1]

        temp_mask = temp_mask.reshape(-1)

        raw_y_r = raw_y_r[temp_mask==0]
        raw_y_g = raw_y_g[temp_mask==0]
        raw_y_b = raw_y_b[temp_mask==0]
        
        dictionary_tr = dictionary_tr[:, temp_mask==0].T
        dictionary_tg = dictionary_tg[:, temp_mask==0].T
        dictionary_tb = dictionary_tb[:, temp_mask==0].T
        
#         for i in range(target_patch[0][0], target_patch[0][1]+1):
#             for j in range(target_patch[1][0], target_patch[1][1]+1):
#                 temp_mask = self.working_mask[i,j]
#                 if not temp_mask:
#                     raw_y_r.append(self.working_image[i,j,0])
#                     raw_y_g.append(self.working_image[i,j,1])
#                     raw_y_b.append(self.working_image[i,j,2])
#                     dictionary_tr.append(self.dictionary_r[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
#                     dictionary_tg.append(self.dictionary_g[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
#                     dictionary_tb.append(self.dictionary_b[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
# #                 else:
# #                     raw_y_r.append(self.working_image[i,j,0])
# #                     raw_y_g.append(self.working_image[i,j,1])
# #                     raw_y_b.append(self.working_image[i,j,2])
# #                     dictionary_tr.append(self.dictionary_r[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
# #                     dictionary_tg.append(self.dictionary_g[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
# #                     dictionary_tb.append(self.dictionary_b[(i - target_patch[0][0]) * self.patch_size + j - target_patch[1][0], :])
    
#         dictionary_tr = np.stack(dictionary_tr, axis=0)
#         dictionary_tg = np.stack(dictionary_tg, axis=0) 
#         dictionary_tb = np.stack(dictionary_tb, axis=0) 
#         raw_y_r = np.array(raw_y_r)
#         raw_y_g = np.array(raw_y_g)
#         raw_y_b = np.array(raw_y_b)
# #         dictionary = np.concatenate((dictionary_tr, dictionary_tg, dictionary_tb), axis=0)
# #         raw_y = np.concatenate((raw_y_r, raw_y_g, raw_y_b), axis=0)
# #         print(dictionary.shape, raw_y.shape)

        self.r_Lasso.fit(dictionary_tr, raw_y_r)
        self.g_Lasso.fit(dictionary_tg, raw_y_g)
        self.b_Lasso.fit(dictionary_tb, raw_y_b)
        
        dictionary_r_p = self.dictionary_r.reshape((-1,self.patch_size, 
                                                    self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w)).T
        dictionary_g_p = self.dictionary_g.reshape((-1,self.patch_size, 
                                                    self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w)).T
        dictionary_b_p = self.dictionary_b.reshape((-1,self.patch_size, 
                                                    self.patch_size))[:, 0:temp_h, 0:temp_w].reshape((-1, temp_h*temp_w)).T
        
        
        predict_r = self.r_Lasso.predict(dictionary_r_p).reshape(
            (temp_h, temp_w))
        predict_g = self.g_Lasso.predict(dictionary_g_p).reshape(
            (temp_h, temp_w))
        predict_b = self.b_Lasso.predict(dictionary_b_p).reshape(
            (temp_h, temp_w))
        
#         self.Lasso.fit(dictionary, raw_y)
#         predict = self.Lasso.predict(self.dictionary)
#         predict_r = predict[:self.patch_size**2].reshape(
#             (self.patch_size, self.patch_size))
#         predict_g = predict[self.patch_size**2:2*self.patch_size**2].reshape(
#             (self.patch_size, self.patch_size))
#         predict_b = predict[2*self.patch_size**2:].reshape(
#             (self.patch_size, self.patch_size))

        # [patch_size, patch_size, 3]
        predict = np.stack([predict_r, predict_g, predict_b], axis=-1)

        return predict

    # def _update_image(self, target_pixel, source_patch):
    def _update_image(self, target_pixel, generated_image):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)

        target_data = self._patch_data(self.working_image, target_patch)

        new_data = generated_image*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def _finished(self):
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0

    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
