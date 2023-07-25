import cv2
import numpy as np

image_path = "socrates.jpg"
socrates_body_path = "socrates_body.jpg"


def find_section_of_image(img_rgb, template):
    w, h = template.shape[:-1]
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    scale_percent = 30  # percent of original size
    width = int(img_rgb.shape[1] * scale_percent / 100)
    height = int(img_rgb.shape[0] * scale_percent / 100)

    cv2.imshow("Detected", cv2.resize(img_rgb, (width, height)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_section_of_image(cv2.imread(image_path), cv2.imread(socrates_body_path))
