import torch
import cv2
import kornia as K
from kornia_moons.viz import draw_LAF_matches


class LoFTR_Matcher:
    # image_size - size of scaled images used in LoFTR, if it is None then the source images
    # or images with the calculated scale are used. For best matching performance,
    # the source image size should be a multiple of image_size.
    def __init__(self, image_size=None, device=None):
        self.image_size = image_size
        self.device = device if device else K.utils.get_cuda_device_if_available()
        self.matcher = K.feature.LoFTR(pretrained='outdoor').eval().to(self.device)

    def __call__(self, image0, image1, confidence_min=0.8, accurate=False):
        image0 = self._convert_image(image0)
        image1 = self._convert_image(image1)

        input = {
            'image0': K.color.rgb_to_grayscale(image0),  # LoFTR works on grayscale images only
            'image1': K.color.rgb_to_grayscale(image1)
        }
        with torch.inference_mode():
            corresp = self.matcher(input)

        # Select key points that have a confidence greater than confidence_min
        mask = corresp['confidence'] > confidence_min
        indices = torch.nonzero(mask, as_tuple=True)
        keypoints0 = corresp['keypoints0'][indices].cpu().numpy()
        keypoints1 = corresp['keypoints1'][indices].cpu().numpy()
        confidence = corresp['confidence'][indices].cpu().numpy()

        # From confidence key points we select inlier points relative to the fundamental matrix
        fmat_method = cv2.USAC_ACCURATE if accurate else cv2.USAC_MAGSAC
        try:
            fmat, inliers = cv2.findFundamentalMat(keypoints0, keypoints1, fmat_method, 1, 0.99, 100000)
            inliers = inliers > 0
        except:
            inliers = None

        return {
            'image0': image0,
            'image1': image1,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'confidence': confidence,
            'inliers': inliers
        }

    @staticmethod
    def draw_matches(match_dict):
        output_fig = draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_dict['keypoints0']).view(1, -1, 2),
                torch.ones(match_dict['keypoints0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_dict['keypoints0'].shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_dict['keypoints1']).view(1, -1, 2),
                torch.ones(match_dict['keypoints1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_dict['keypoints1'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(match_dict['keypoints0'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(match_dict['image0']),
            K.tensor_to_image(match_dict['image1']),
            match_dict['inliers'],
            draw_dict={
                'inlier_color': (0.2, 1, 0.2),
                'tentative_color': (1, 0.1, 0.1),
                'feature_color': (0.2, 0.5, 1),
                'vertical': False}
        )
        return output_fig

    def _convert_image(self, image):
        SIZE_MAX = 1280  # LoFTR has out of memory for large images
        image = K.utils.image_to_tensor(image)
        image = image.float().unsqueeze(dim=0).to(self.device) / 255.0
        if self.image_size:
            image = K.geometry.resize(image, self.image_size, interpolation='area')
        elif max(image.shape[-1], image.shape[-2]) > SIZE_MAX:
            image = K.geometry.resize(image, SIZE_MAX, side='long', interpolation='area')
        return image