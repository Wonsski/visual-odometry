import cv2
import numpy as np
import os 

class VisualOdometry:

    def __init__(self):
        self.orb = cv2.ORB_create(10000)  
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 

        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None


        # KITTI camera calibration
        self.K = np.array([[718.856, 0, 607.1928],
                           [0, 718.856, 185.2157],
                           [0, 0, 1]])
        calib_file = "data/calib.txt"
        with open(calib_file, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            self.K = P[0:3, 0:3]
        
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory_image = np.zeros((1200, 1200, 3), dtype=np.uint8) + 255
    
    def load_kitti(self, data_dir):
        """

        Loads all files from ``data_dir``

        """
        img_data = os.path.join(data_dir, "image_0/")

        img_files = os.listdir(img_data)
    
        images = [cv2.imread(os.path.join(img_data, img), cv2.IMREAD_GRAYSCALE) for img in sorted(img_files)]
        
        return images
    
    def find_keypoints(self, img):
        """
        
        Finds keypoints and descriptors in ``img`` uding self.orb
        
        """
        keypoints, descriptors = self.orb.detectAndCompute(img, None)

        return keypoints, descriptors
    

    def match_keypoints(self, descriptors1, descriptors2):
        """
        
        Matches descriptors ``descriptors1``, ``descriptors1``
        
        """

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []

        ratio_thresh = 0.75
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def estimate_pose(self, keypoints1, keypoints2, matches):
        """

        Calculates rotation ``R`` aand translation ``t`` matrices

        """
        
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t
    
    def draw_keypoints(self, img, keypoints):
        """

        Draws ``keypoints`` on the ``img``

        """
        return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """
        
        Draws matches between ``img1`` and ``img2`` using kyepoints ``kp1``, ``kp2`` and ``matches`` list 

        """
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_img
    
    def draw_trajectory(self):
        """
        
        Draws trajectory on self.trajetory_image

        """
        x, y = int(self.t[0]) + 600, int(self.t[2]) + 600
        cv2.circle(self.trajectory_image, (x, y), 2, (0, 0, 255), 1)


    def play_from_list(self, images):
        for image in images:
            if image is None:
                continue

            keypoints, descriptors = self.find_keypoints(image)

            if self.prev_image is not None:
                matches = self.match_keypoints(self.prev_descriptors, descriptors)
                R, t = self.estimate_pose(self.prev_keypoints, keypoints, matches)

                img_kp1 = self.draw_keypoints(self.prev_image, self.prev_keypoints)
                img_kp2 = self.draw_keypoints(image, keypoints)
                img_matches = self.draw_matches(self.prev_image, self.prev_keypoints, image, keypoints, matches)
                
                cv2.imshow("Matches", img_matches)


                self.t += self.R @ t 
                self.R = R @ self.R   


                self.draw_trajectory()
                cv2.imshow("Trajectory", self.trajectory_image)

            self.prev_image = image
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors

            cv2.imshow("Current Image", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    
    vo = VisualOdometry()

    images = vo.load_kitti("data/dataset/sequences/06/")

    vo.play_from_list(images)

if __name__=="__main__":
    main()