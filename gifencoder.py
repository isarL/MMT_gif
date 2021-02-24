import os
import argparse
import numpy as np
import lzw
from sklearn import cluster

def calculate_psnr(img_A, img_B):
    n, m, _= img_A.shape
    MSE = np.sum((1/(m*n*3))*np.square(img_A-img_B)) #als ik (1/(m*n*3)) erbuiten zet krijg ik soms negatieve getallen, ik denk door overflow
    return 10*np.log10(255*255/MSE)

class GIFEncoder:

    def __init__(self, img, bitdepth):

        # The input image
        self.img = np.array(img).astype(np.int16)
        self.img_dims = self.img.shape

        assert len(self.img_dims) == 3, 'Image needs to have three color dimensions'

        # The LUT table
        self.color_table = None
        self.color_table_bits = bitdepth           # max 8
        self.color_table_size = 1 << bitdepth      # max 256

        # The LUT indices per pixel (which go into the bitstream)
        self.color_table_indices = None

        # The reconstructed image (for calculting PSNR)
        self.img_coded = None

    def make_random_color_tabel(self):
        self.color_table = np.random.randint(0, 256, (self.color_table_size, 3), dtype=np.uint8)

    def make_grayscale_color_table(self):
        self.color_table = np.array([[i, i, i] for i in range(0, 256, 255//(self.color_table_size-1))],   dtype=np.uint8)

    def make_random_sample_color_table(self):
        self.color_table = np.array([self.img[np.random.randint(0, self.img_dims[0]), np.random.randint(0, self.img_dims[1])] for i in range(0, self.color_table_size)] ,   dtype=np.uint8)

    def make_median_cut_color_table(self):
        r = (np.array(self.img[:,:,0])).flatten()
        r.sort()
        g = (np.array(self.img[:,:,1])).flatten()
        g.sort()
        b = (np.array(self.img[:,:,2])).flatten()
        b.sort()
        blok = np.array([r, g, b])
        def divide(r, g, b, d):
            if d == 1:
                return [(r[len(r)//2], g[len(g)//2], b[len(b)//2])]
            r_range = r[len(r)-1]- r[0]
            g_range = g[len(g)-1] - g[0]
            b_range = b[len(b)-1] - b[0]
            if r_range >= g_range and r_range >= b_range:
                l = divide(r[:len(r)//2], g, b, d//2)
                l.extend(divide(r[len(r)//2:], g, b, d//2))
                return l
            if g_range >= r_range and g_range >= b_range:
                l = divide(r, g[:len(g)//2], b, d//2)
                l.extend(divide(r, g[len(g)//2:], b, d//2))
                return l
            if b_range >= g_range and b_range >= r_range:
                l = divide(r, g, b[:len(g)//2], d//2)
                l.extend(divide(r, g, b[len(b)//2:], d//2))
                return l
        self.color_table = np.array(divide(r, g, b, self.color_table_size))

    def make_kmeans_color_table(self):
        raise NotImplementedError()

    def find_nearest_color_index(self, rgb_vec):
        distances = np.square(
                self.color_table.astype(np.float) -
                np.expand_dims(rgb_vec.astype(np.float), axis=-2))
        distances = np.sum(distances, axis=-1)
        return np.argmin(distances, axis=-1)

    def transform_image(self, dithering):
        if dithering == 0:
            self.color_table_indices = np.zeros((self.img_dims[0],
                self.img_dims[1]), dtype=np.uint8)
            self.img_coded = np.zeros(self.img_dims, dtype=np.uint8)
            for row in range(self.img_dims[0]):
                self.color_table_indices[row] = self.find_nearest_color_index(self.img[row])
                self.img_coded[row] = self.color_table[self.color_table_indices[row]]
        elif dithering == 1:
            raise NotImplementedError()
        elif dithering == 2:
            raise NotImplementedError()
        else:
            raise Exception('Dithering method should be 0, 1, or 2')


    def encode(self, output_path):
        with open(output_path, 'wb') as w:
            # "GIF89a" in Hex
            w.write(bytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]))

            # width and height in unsigned 2 byte (16 bit) little-endian
            width_bytes = (self.img.shape[1]).to_bytes(2, byteorder='little')
            height_bytes = (self.img.shape[0]).to_bytes(2, byteorder='little')
            w.write(width_bytes)
            w.write(height_bytes)

            # GCT follows for 256 colors with resolution 3 x 8 bits/primary;
            # the lowest 3 bits represent the bit depth minus 1, the highest
            # true bit means that the GCT is present
            w.write(bytes([0xf0 + self.color_table_bits - 1]))

            # Background color #0
            w.write(bytes([0x00]))
            # Default pixel aspect ratio
            w.write(bytes([0x00]))

            # Global color table (GCT)
            assert self.color_table_size == self.color_table.shape[0]
            for c in range(self.color_table_size):
                r,g,b = self.color_table[c]
                w.write(bytes([r, g, b]))

            # Graphic Control Extension (comment fields precede this in most files)
            w.write(bytes([0x21, 0xf9, 0x03, 0x00, 0x00, 0x00, 0x00]))

            # Image Descriptor
            w.write(bytes([0x2c]))
            w.write(bytes([0x00, 0x00, 0x00, 0x00])) # NW corner position of image in logical screen
            w.write(width_bytes)
            w.write(height_bytes)

            w.write(bytes([0x00])) # no local color table
            lzw_min = max(2, self.color_table_bits)
            max_code_size = 10

            # start of image - LZW minium
            w.write(lzw_min.to_bytes(1, byteorder='little'))

            color_table_indices = ''.join([chr(x) for x in self.color_table_indices.flatten()])
            compressed_indices = lzw.compress(color_table_indices, lzw_min, max_code_size)

            for i, byte in enumerate(compressed_indices):
                if i % 255 == 0:
                    # Write length of coded stream in bytes (subblock can maximum be 255 long)
                    w.write((min(255, len(compressed_indices)-i)).to_bytes(1, byteorder='little'))
                w.write(byte.to_bytes(1, byteorder='little'))

            w.write(bytes([0x00, 0x3b])) # end of image data, end of GIF file

from PIL import Image
import argparse

def main(image_path, output_path, bitdepth, lut_method, dithering):
    '''
        Do NOT change this function. Only complete the functions that raise
        NotImplementedError().  You can make additional member functions if
        necessary.
    '''
    assert bitdepth == None or (bitdepth <= 8 and bitdepth >= 1), 'bitdepth needs to be in range [1,8]'

    if lut_method is None:
        lut_method = 'random-colors'

    if dithering is None:
        dithering = 0

    if bitdepth is None:
        bitdepth = 6

    im = Image.open(image_path).convert("RGB")
    im = np.array(im)
    if im.dtype == np.bool:
        im = im.astype(np.int16) * 255
    else:
        im = im.astype(np.int16)
    if len(im.shape) == 2:
        im = np.expand_dims(im, -1)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, -1)
    print(im.shape)

    enc = GIFEncoder(im, bitdepth)

    print("make color table...")
    if lut_method == 'grayscale':
        enc.make_grayscale_color_table()
    elif lut_method == 'random-colors':
        enc.make_random_color_tabel()
    elif lut_method == 'random-sampling':
        enc.make_random_sample_color_table()
    elif lut_method == 'median-cut':
        enc.make_median_cut_color_table()
    elif lut_method == 'kmeans':
        enc.make_kmeans_color_table()
    else:
        raise ValueError("Unknown lut method " + str(lut_method))
    assert enc.color_table_size == enc.color_table.shape[0]

    print("transform...")
    enc.transform_image(dithering)
    print("encode...")
    enc.encode(output_path)
    print("psnr...")
    psnr = calculate_psnr(enc.img, enc.img_coded)

    print('PSNR: %6.3f dB' % psnr)


if __name__ == '__main__':
    
    class CustomHelpFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings) + '   ' + args_string

    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument('-i', '--image-path', type=str, required=True, help="input image")
    parser.add_argument('-o', '--output-path', type=str, required=True, help="output_file")
    parser.add_argument('-b', '--bitdepth', type=int, required=False, help="bitdepth")
    parser.add_argument('-m', '--lut-method', type=str, required=False,
            choices=['grayscale', 'random-colors', 'random-sampling', 'median-cut', 'kmeans'],
            help="LUT method")
    parser.add_argument('-d', '--dithering', type=int, required=False,
            choices=[0, 1, 2],
            help="dithering method (0 = no dithering, 1 = Floyd-Steinberg, 2 = MAE)")
    
    args = parser.parse_args()

    main(**vars(args))
    end = input("to close press enter")
    
