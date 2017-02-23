from PIL import Image
import sys
im1=Image.open(sys.argv[1])
im2=Image.open(sys.argv[2])
width, height=im1.size
im = Image.new("RGBA",(width, height),"white")
for x in range(width):
    for y in range(height):
        if im1.getpixel((x,y))[0] == im2.getpixel((x,y))[0] and im1.getpixel((x,y))[1] == im2.getpixel((x,y))[1] and im1.getpixel((x,y))[2] == im2.getpixel((x,y))[2] and im1.getpixel((x,y))[3] == im2.getpixel((x,y))[3]:
            pix = (0,0,0,0)
            im.putpixel((x,y),pix)
        else:
            im.putpixel((x,y),im2.getpixel((x,y)))
im.save('ans_two.png')
