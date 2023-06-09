from tkinter import Tk, filedialog

################################################################################
def prompt_select_file():
    root = Tk()
    root.withdraw()
    filePath = filedialog.askopenfilename(
        initialdir = root,
        title = "Choose an image...",
        filetypes = [ # filetypes supported by Pillow
            ("All files", "*.*"),
            ("BLP", "*.blp"),
            ("BMP image/bmp", "*.bmp"),
            ("BUFR", "*.bufr"),
            ("CUR", "*.cur"),
            ("DCX", "*.dcx"),
            ("DDS", "*.dds"),
            ("DIB image/bmp", "*.dib"),
            ("EPS application/postscript", "*.eps *.ps"),
            ("FITS", "*.fit *.fits"),
            ("FLI", "*.flc *.fli"),
            ("FTEX", "*.ftc *.ftu"),
            ("GBR", "*.gbr"),
            ("GIF image/gif", "*.gif"),
            ("GRIB", "*.grib"),
            ("HDF5", "*.h5 *.hdf"),
            ("ICNS image/icns", "*.icns"),
            ("ICO image/x-icon", "*.ico"),
            ("IM", "*.im"),
            ("IPTC", "*.iim"),
            ("JPEG image/jpeg", "*.jfif *.jpe *.jpeg *.jpg"),
            ("JPEG2000 image/jp2", "*.j2c *.j2k *.jp2 *.jpc *.jpf *.jpx"),
            ("MPEG video/mpeg", "*.mpeg *.mpg"),
            ("MSP", "*.msp"),
            ("PCD", "*.pcd"),
            ("PCX image/x-pcx", "*.pcx"),
            ("PIXAR", "*.pxr"),
            ("PNG image/png", "*.apng *.png"),
            ("PPM image/x-portable-anymap", "*.pbm *.pgm *.pnm *.ppm"),
            ("PSD image/vnd*.adobe*.photoshop", "*.psd"),
            ("SGI image/sgi", "*.bw *.rgb *.rgba *.sgi"),
            ("SUN", "*.ras"),
            ("TGA image/x-tga", "*.icb *.tga *.vda *.vst"),
            ("TIFF image/tiff", "*.tif *.tiff"),
            ("WEBP image/webp", "*.webp"),
            ("WMF", "*.emf *.wmf"),
            ("XBM image/xbm", "*.xbm"),
            ("XPM image/xpm", "*.xpm"),
        ]
    )
    root.destroy()
    return filePath

################################################################################
