import cv2

print("Welcome to the coding starter!")

# Variables
image_path = r"C:\Users\JeremiahFallin\dev\summer\neural-network-ninjas\socrates.jpg"

# Basic Input/Output
print("The path to the image is:", image_path)

answer = input("Do you want to display the image? (yes/no): ")

if answer.lower() == "yes":
    print("Great! Let's display the image.")
elif answer.lower() == "no":
    print("Alright, we won't display the image.")
else:
    print("Invalid input. Please enter 'yes' or 'no'.")


def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def display_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the image
image = cv2.imread(image_path)

scale_percent = 30  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
print("Resizing image to", width, "x", height)
image = cv2.resize(image, (width, height))

if answer.lower() == "yes":
    # Display the original image
    display_image(image, "Original Image")

    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Display the blurred image
    display_image(blurred_image, "Blurred Image")

    # Apply a threshold to the image
    threshold_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY)[1]

    # Display the thresholded image
    display_image(threshold_image, "Thresholded Image")

    # Convert the image to grayscale
    gray_image = convert_to_grayscale(image)

    # Display the grayscale image
    display_image(gray_image, "Grayscale Image")

    # Invert the grayscale image
    invert_image = cv2.bitwise_not(gray_image)

    # Display the inverted image
    display_image(invert_image, "Inverted Image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
