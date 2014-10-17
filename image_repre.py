from PIL import Image
import random


def openGrayScale(imageName):
    # opens an image in gray scale mode
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


def plot8x8(grayscale_list, show=False):
    image = Image.new("L", (8, 8))
    image.putdata(grayscale_list)
    if show:
        image.show()
    return image


def visualize_weight(weight, show=False):
    # the weight should be an numpy array
    numNeuron, numInput = weight.shape
    weightList = weight.tolist()
    # normalize the weights of each neuron
    for i in range(len(weightList)):
        neuron_weight = weightList[i]
        mx = max(neuron_weight)
        weightList[i] = [w*255/mx for w in neuron_weight]
    print weightList

    imageList = [plot8x8(neuron) for neuron in weightList]
    bigImage = Image.new("L", (64, 64))
    for i in range(len(imageList)):
        im = imageList[i]
        x, y = i//5, i % 5
        bigImage.paste(im, (4+12*x, 4+12*y, 12+12*x, 12+12*y))
    if show:
        bigImage.show()


def main():
    name = "reus.png"
    image = openGrayScale(name)
    plot8x8(randomSelect8x8(image))

if __name__ == '__main__':
    main()
