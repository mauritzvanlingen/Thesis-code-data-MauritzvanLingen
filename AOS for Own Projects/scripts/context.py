import cv2
import os

# Path to the directory where the processed images will be saved
#output_path = r"C:\Users\Beheerder\Documents\AI bachelor documenten\jaar 3, semester 2\Thesis\AOS-stable_release\AOS-stable_release\LFR\data\F0\images_ldr_gaussian_blurred"

# Loop through all files in the input directory'

def image_processor(method):
    input_path = r"C:\Users\Beheerder\Documents\AI bachelor documenten\jaar 3, semester 2\Thesis\AOS-stable_release\AOS-stable_release\AOS integrated images N = 20"
    if method == "Gaussian":
        output_path = r"C:\Users\Beheerder\Documents\AI bachelor documenten\jaar 3, semester 2\Thesis\AOS-stable_release\AOS-stable_release\AOS integrated images Gaussian blurred"

        for filename in os.listdir(input_path):
    # Load the image from file

                image = cv2.imread(os.path.join(input_path, filename))

    # Apply a Gaussian filter to the image
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
  
    # Save the processed image to file
                cv2.imwrite(os.path.join(output_path, filename), blurred)
            
    if method == "CLAHE":
        output_path = r"C:\Users\Beheerder\Documents\AI bachelor documenten\jaar 3, semester 2\Thesis\AOS-stable_release\AOS-stable_release\AOS integrated images CLAHE"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for filename in os.listdir(input_path):
    # Load the image from file
                image = cv2.imread(os.path.join(input_path, filename))
            
        
                clahe_image = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                cv2.imwrite(os.path.join(output_path, filename), clahe_image)
    if method == "AHE":
        output_path = r"C:\Users\Beheerder\Documents\AI bachelor documenten\jaar 3, semester 2\Thesis\AOS-stable_release\AOS-stable_release\AOS integrated images AHE"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        for filename in os.listdir(input_path):
            # Load the image from file
           
                image = cv2.imread(os.path.join(input_path, filename))
               
            # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply AHE to the grayscale image
                ahe = cv2.equalizeHist(gray)

            # Save the processed image to file
                cv2.imwrite(os.path.join(output_path, filename), ahe)


image_processor("AHE")




