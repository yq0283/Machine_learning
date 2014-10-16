from PIL import Image
import random


def openGrayScale(imageName):
    # opens an image in gray scale mode
    # try:
    #     return Image.open(imageName).convert("L")
    # except:
    #     return None
    return Image.open(imageName).convert("L")


def randomSelect8x8(image):
    # randomly select an 8x8 square in the image
    # image parameter should be in grayscale
    # returns a list of grayscales of pixels
    l = []
    x = random.randint(0, image.size[0]-8)
    y = random.randint(0, image.size[1]-8)
    for xi in range(x, x+8):
        for yi in range(y, y+8):
            l.append(image.getpixel((xi, yi)))
    return l


def plot8x8(grayscale_list):
    image = Image.new("L", (8, 8))
    image.putdata(grayscale_list)
    image.show()


def main():
    name = "reus.png"
    image = openGrayScale(name)
    plot8x8(randomSelect8x8(image))

if __name__ == '__main__':
    main()
