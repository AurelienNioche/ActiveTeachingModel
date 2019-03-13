from PIL import Image, ImageDraw, ImageFont


def create_image_from_character(character, size=28, name=None, folder='auto_encoder/image',
                                font='auto_encoder/font/arialunicodems.ttf'):

    text_color = (255, 255, 255)

    background_color = (0, 0, 0)

    img = Image.new('RGB', (size, size), color=background_color)

    fnt = ImageFont.truetype(font, int(size*2/3))
    d = ImageDraw.Draw(img)
    w, h = d.textsize(character, font=fnt)
    d.text(((size-w)/2, (size-h)/2 - h/10), character, font=fnt, fill=text_color)

    if name is None:
        name = ord(character)

    img.save(f"{folder}/{name}.png")


if __name__ == "__main__":

    word_list = ['七', '中', '二', '八', '力', '王', '生', '花', '虫', '足']

    for word in word_list:
        create_image_from_character(word, 50)
