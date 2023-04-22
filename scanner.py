import cv2
import imutils
from skimage.filters import threshold_local
from transform import perspective_transform #a user module named transform

import tkinter as tk
from tkinter import filedialog

# function to handle image upload
def upload_image():
    # get file path from user using file dialog
    file_path = filedialog.askopenfilename()
    if file_path:
        # load image and continue with scanning process
        org_img = cv2.imread(file_path)
        copy = org_img.copy()
        # resized height in hundreds
        ratio = org_img.shape[0] / 500.0
        img_resize = imutils.resize(org_img, height=500)

        # displaying output
        cv2.imshow('Resized image', img_resize)

        # waiting for user to press any key
        cv2.waitKey(0)

        # converting to grayscale
        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayed image', gray)
        cv2.waitKey(0)

        # applying edge detector
        blurr_image = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_image = cv2.Canny(blurr_image, 75, 200)
        cv2.imshow('Image edges', edge_image)
        cv2.waitKey(0)

        # finding the largest contour
        cnts, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        doc = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                doc = approx
                break

        # check if a contour with four corners was found
        if doc is None:
            print("No contour with four corners found")
        else:
            # circling the four corners
            p = []

            for d in doc:
                tuple_point = tuple(d[0])
                cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
                p.append(tuple_point)
            cv2.imshow('Circled corner points', img_resize)
            cv2.waitKey(0)

            warped_image = perspective_transform(copy, doc.reshape(8, 2) * ratio)
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Warped Image", imutils.resize(warped_image, height=650))
            cv2.waitKey(0)

            # applying adaptive threshold and saving the scanned output
            T = threshold_local(warped_image, 11, offset=10, method="gaussian")
            warped = (warped_image > T).astype("uint8") * 255
            cv2.imwrite('./' + 'scan' + '.png', warped)

            # displaying output
            cv2.imshow("Final Scanned image", imutils.resize(warped, height=650))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# create GUI window
window = tk.Tk()
window.title("Image Scanner")

# create upload button
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# run GUI loop
window.mainloop()

