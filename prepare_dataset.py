import os
import shutil

# Original Food-101 images
SOURCE = "images"

# Destination folders
HEALTHY = "Healthy"
UNHEALTHY = "Unhealthy"

# Create folders if they don't exist
os.makedirs(HEALTHY, exist_ok=True)
os.makedirs(UNHEALTHY, exist_ok=True)

# Generally more suitable food choices
healthy_foods = [
    "beet_salad",
    "caprese_salad",
    "ceviche",
    "greek_salad",
    "grilled_salmon",
    "guacamole",
    "hummus",
    "edamame",
    "mussels",
    "oysters",
    "omelette"
]

# Foods that are generally better limited
unhealthy_foods = [
    "apple_pie",
    "baklava",
    "beignets",
    "bread_pudding",
    "cannoli",
    "cheesecake",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "cup_cakes",
    "donuts",
    "french_fries",
    "ice_cream",
    "macarons",
    "nachos"
]


def copy_images(food_list, destination):
    count = 0

    for food in food_list:
        folder = os.path.join(SOURCE, food)

        if not os.path.isdir(folder):
            print("Folder not found:", food)
            continue

        for file in os.listdir(folder):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                source_file = os.path.join(folder, file)

                # Add food name to avoid duplicate filenames
                new_name = food + "_" + file
                destination_file = os.path.join(destination, new_name)

                # Copy only if the image is not already present
                if not os.path.exists(destination_file):
                    shutil.copy2(source_file, destination_file)
                    count += 1

    print("Copied", count, "images to", destination)


# Copy images
copy_images(healthy_foods, HEALTHY)
copy_images(unhealthy_foods, UNHEALTHY)

print("\nDataset preparation completed!")