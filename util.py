
# import colorsys
# import math
# import operator
# from PIL import Image, ImageChops
# import functools
# from math import sqrt
# import numpy as np
# from itertools import chain


# def rmsdiff(im1, im2):
#     im1 = im1.convert("RGBA")
#     im2 = im2.convert("RGBA")

#     diff = ImageChops.difference(im1, im2)
#     h = diff.histogram()

#     blah = np.array(h, dtype=int)
#     okay = (np.arange(1024) % 256)**2
#     rms = sqrt(np.sum(blah*okay)/float(im1.size[0] * im1.size[1]))

#     return rms


# def get_multi(im, min_size):
#     w, h = im.size
#     if w > h:
#         return float(min_size)/w
#     return float(min_size)/h


# def restrain_img_size(im, max_pix=1700):
#     max_size = (max_pix, max_pix)
#     w, h = im.size
#     if w > max_pix or h > max_pix:
#         im.thumbnail(max_size, Image.ANTIALIAS)

#     return im


# def mult_img_size(im, scale):
#     w, h = int(im.size[0]*scale), int(im.size[1]*scale)
#     if scale > 1:
#         im = im.resize((w, h), Image.ANTIALIAS)
#     else:
#         im.thumbnail((w, h), Image.ANTIALIAS)

#     return im


# def enlarge_img(im, max_pix=9000):
#     w, h = im.size
#     if w < max_pix and h < max_pix:
#         if w > h:
#             m = float(max_pix)/w
#             h = int(m*h)
#             w = max_pix

#             resize = (w, h)
#         else:
#             m = float(max_pix)/h
#             w = int(m*w)
#             h = max_pix
#             resize = (w, h)
#         im = im.resize(resize, Image.ANTIALIAS)

#     return im


# def png_to_jpeg(im):
#     im=im.convert('RGB')
#     og_image_rgb = Image.new("RGB", im.size, (255,255,255))
#     og_image_rgb.paste(im)
#     return og_image_rgb


# def clamp_int(val, minval, maxval):
#     if val < minval:
#         return int(minval)
#     if val > maxval:
#         return int(maxval)

#     return int(val)


# def image_transpose_exif(im):
#     exif_orientation_tag = 0x0112  # contains an integer, 1 through 8
#     exif_transpose_sequences = [   # corresponding to the following
#         [],
#         [Image.FLIP_LEFT_RIGHT],
#         [Image.ROTATE_180],
#         [Image.FLIP_TOP_BOTTOM],
#         [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
#         [Image.ROTATE_270],
#         [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],
#         [Image.ROTATE_90],
#     ]

#     try:
#         seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag] - 1]
#     except Exception:
#         return im
#     else:
#         return functools.reduce(lambda im, op: im.transpose(op), seq, im)


# def average_color_img(image):
#     result = image.quantize(colors=1).convert('RGB').getcolors()
#     return result[0][1]

# # deprecated in favor of average_color_img
# # def average_color(image, rect=None):
# #     if not rect:            # Use whole image
# #         w, h = image.size
# #         x0, y0 = (0, 0)
# #         x1, y1 = (w, h)
# #     else:                   # Use subset rect of image
# #         x0, y0, x1, y1 = rect
# #         w = abs(x0 - x1)
# #         h = abs(y0 - y1)

# #     r, g, b = 0, 0, 0
# #     area = w*h

# #     for x in range(x0, x1):
# #         for y in range(y0, y1):
# #             cr, cg, cb = image.getpixel((x, y))
# #             r += cr
# #             g += cg
# #             b += cb

# #     return (r/area, g/area, b/area)


# # DEG30 = 30/360.
# # def adjacent_colors((r, g, b), d=DEG30):  # Assumption: r, g, b in [0, 255]
# #     r, g, b = map(lambda x: x/255., [r, g, b])  # Convert to [0, 1]
# #     h, l, s = colorsys.rgb_to_hls(r, g, b)      # RGB -> HLS
# #     h = [(h+d) % 1 for d in (-d, d)]            # Rotation by d

# #     adjacent = [map(lambda x: int(round(x*255)), colorsys.hls_to_rgb(hi, l, s))
# #             for hi in h]  # H'LS -> new RGB

# #     adjacent[0] = tuple(adjacent[0])
# #     adjacent[1] = tuple(adjacent[1])
# #     return adjacent

# # rgb_scale = 255
# # cmyk_scale = 100
# # def rgb_to_cmyk(r, g, b):
# #     if (r == 0) and (g == 0) and (b == 0):
# #         # black
# #         return 0, 0, 0, cmyk_scale

# #     # rgb [0,255] -> cmy [0,1]
# #     c = 1 - r / float(rgb_scale)
# #     m = 1 - g / float(rgb_scale)
# #     y = 1 - b / float(rgb_scale)

# #     # extract out k [0,1]
# #     min_cmy = min(c, m, y)
# #     c = (c - min_cmy)
# #     m = (m - min_cmy)
# #     y = (y - min_cmy)
# #     k = min_cmy

# #     # rescale to the range [0,cmyk_scale]
# #     return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale


# # def cmyk_to_rgb(c, m, y, k):
# #     """
# #     """
# #     r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
# #     g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
# #     b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
# #     return int(r),int(g),int(b)


# # def hilo(a, b, c):
# #     if c < b: b, c = c, b
# #     if b < a: a, b = b, a
# #     if c < b: b, c = c, b
# #     return a + c


# # def complement(r, g, b):
# #     k = hilo(r, g, b)
# #     return tuple(k - u for u in (r, g, b))

# # # http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
# # # third option half way page down
# # def luminance(r, g, b):
# #     return math.sqrt(0.299 * math.pow(r, 2) + 0.587 * math.pow(g, 2) + 0.114 * math.pow(b, 2))


# # def tint_to_lum(color, lum):
# #     r, g, b = color
# #     nR, nG, nB = r, g, b
# #     while True:
# #         tint_factor = .005

# #         nR = nR + (255 - nR) * tint_factor
# #         nG = nG + (255 - nG) * tint_factor
# #         nB = nB + (255 - nB) * tint_factor
# #         if luminance(nR, nG, nB) >= lum:
# #             break
# #     return int(nR), int(nG), int(nB)


# # # factor in all other lums in color
# # def tint_to_lums(color, base_colors, lum):
# #     r, g, b = color
# #     nR, nG, nB = r, g, b
# #     lum_total = 0
# #     for col in base_colors:
# #         lum_total += luminance(col[0], col[1], col[2])

# #     while True:
# #         tint_factor = .005
# #         nR = nR + (255 - nR) * tint_factor
# #         nG = nG + (255 - nG) * tint_factor
# #         nB = nB + (255 - nB) * tint_factor
# #         if (luminance(nR, nG, nB) + lum_total)/(len(base_colors)+1) >= lum or (luminance(nR, nG, nB) >= 254):
# #             break

# #     return int(nR), int(nG), int(nB)


# # # naive.... need to refactor
# # def shade_to_lum(color, lum):
# #     r, g, b = color
# #     nR, nG, nB = r, g, b
# #     while True:
# #         shade_factor = .005
# #         nR = nR * (1 - shade_factor)
# #         nG = nG * (1 - shade_factor)
# #         nB = nB * (1 - shade_factor)

# #         if luminance(nR, nG, nB) <= lum:
# #             break
# #     return int(nR), int(nG), int(nB)
