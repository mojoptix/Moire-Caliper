# -*- coding: utf-8 -*-
"""
[EN]  ---   MOIRE CALIPER
[FR]  ---   PIED A COULISSE A MOIRE
-----------------------------------------------------------------------------
      Author: Mojoptix
      Website: www.mojoptix.com
      Email: julldozer@mojoptix.com
      Date: 02 march 2018
      License: MIT License
      
      Copyright (c) 2018 Mojoptix
-----------------------------------------------------------------------------
[EN]  This script can be used to generate Moire images to build a Moire Caliper.
      An episode of Mojoptix describes the Moire Caliper in details:
      http://www.mojoptix.com/?p=178

[FR]  Ce script permet de generer des images Moires pour construire un Pied a 
      Coulisse a Moire.
      Un episode de MakerLambda (la version francophone de Mojoptix) decrit le
      Pied a Coulisse a Moire en details:
      http://www.mojoptix.com/?p=182
          
"""




import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import matplotlib.pyplot as plt


# parameters
img_dpi = 2400
my_font_filename = "Carlito-Bold.ttf" # this is an open source font from Google that is equivalent to Calibri: https://www.ecsoft2.org/carlito-fonts 
#my_font_filename = "calibrib.ttf" # uncomment this line if you'd prefer to use the Windows font

# create triangular moire image
def create_triangular_moire_img(img_size_mm=[120.0,20.0], pitch_mm=2.0, line_thickness_mm1=1.9, line_thickness_mm2=1.0, offset_mm=0.0):
    '''Create a moire pattern made of black vertical lines with a line thickness that is changing from bottom to top.
    Input:
        img_size_mm: [mm] size of the resulting image,
        pitch_mm: [mm] distance between the center of the lines,
        line_thickness_mm1: [mm] line thickness at the bottom of the image, 
        line_thickness_mm2: [mm] line thickness at the top of the image, 
        offset_mm: [mm] translate the moire pattern by this amount.
    return:
        numpy array for the moire image.
    '''  
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)               # set the image to all transparent/white
    line_thickness_mm = np.linspace(line_thickness_mm1, line_thickness_mm2, img.shape[0])
    for xx in np.arange(img.shape[0]):
        line_th = np.max([0.0, line_thickness_mm[xx]])
        line_th = np.min([pitch_mm, line_thickness_mm[xx]])
        Tyy = np.arange(img.shape[1])*25.4/img_dpi        # pixel y-coordinates in [mm] across the image
        tmp = np.mod( Tyy-offset_mm , pitch_mm)           # pixel y-coordinates in [mm] across each line pair (eg: {dark line + transparent line})
        tmp_min = 0.5*pitch_mm-0.5*line_th  # pixel y-coordinates in [mm] of the left side of the dark line (inside a line pair)
        tmp_max = 0.5*pitch_mm+0.5*line_th  # pixel y-coordinates in [mm] of the right side of the dark line (inside a line pair)
        tmp2 = np.logical_and( tmp>tmp_min, tmp<tmp_max ) # find all the pixels in between
        img[xx,np.where(tmp2)] = 0                        # and turn them to black
    return img

# create moire scale
def create_moire_scale_img(img_size_mm=[60.0,5.0], scale_start_mm=5.0, scale_end_mm=55.0, font_size=0.15*img_dpi, tick_thickness_mm=0.1, text_height_mm=0.0):
    '''Create the scale for a moire (100 tick marks).
    Input:
        img_size_mm: [mm] size of the image,
        scale_start_mm: [mm] distance from the left side of the image to the 1st tick,
        scale_end_mm: [mm] distance from the right side of the image to the last tick,
        font_size: [pixels] size of the font,
        tick_thickness_mm: [mm] width of the tick marks, 
    Ouput:
        numpy array for the image.
    '''
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    scale_start_pixel = int(scale_start_mm*img_dpi/25.4)
    scale_end_pixel = int(scale_end_mm*img_dpi/25.4)
    tick_center_pixel = np.linspace(scale_start_pixel, scale_end_pixel, 101).astype(int)
    # ADD THE TEXT
    text_height_pixel = int(text_height_mm*img_dpi/25.4)
    font_filename = my_font_filename
    font_size=int(font_size)
    ## Create a temporary PIL image to work on the text
    im = Image.fromarray(img.astype(np.uint8), mode="L") # array filled with 255
    font = ImageFont.truetype(font_filename, font_size)
    draw = ImageDraw.Draw(im)
    ## write the numbers
    draw.text((tick_center_pixel[0]-font.getsize("0")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"0",0,font=font)    
    draw.text((tick_center_pixel[10]-font.getsize("1")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"1",0,font=font)
    draw.text((tick_center_pixel[20]-font.getsize("2")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"2",0,font=font)
    draw.text((tick_center_pixel[30]-font.getsize("3")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"3",0,font=font)
    draw.text((tick_center_pixel[40]-font.getsize("4")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"4",0,font=font)    
    draw.text((tick_center_pixel[50]-font.getsize("5")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"5",0,font=font)
    draw.text((tick_center_pixel[60]-font.getsize("6")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"6",0,font=font)
    draw.text((tick_center_pixel[70]-font.getsize("7")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"7",0,font=font)
    draw.text((tick_center_pixel[80]-font.getsize("8")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"8",0,font=font)
    draw.text((tick_center_pixel[90]-font.getsize("9")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"9",0,font=font)
    draw.text((tick_center_pixel[100]-font.getsize("10")[0]/2, int(0.25*img.shape[0]-text_height_pixel)),"10",0,font=font)
    ## apply the text to the numpy array
    img[np.where(np.asarray(im)==0)] = 0
    # ADD THE TICK MARKS
    tick_height_pixel = 0.15*img.shape[0]*np.ones(101)  # the mini ticks
    tick_height_pixel[5*np.arange(21)] = 0.2*img.shape[0] # every 5 ticks
    tick_height_pixel[10*np.arange(11)] = 0.25*img.shape[0] # every 10 ticks
    tick_thickness_pixel = tick_thickness_mm*img_dpi/25.4 *np.ones_like(tick_height_pixel) # the mini ticks
    tick_thickness_pixel[5*np.arange(21)] = 1.5* tick_thickness_mm*img_dpi/25.4 # every 5 ticks
    for ii in np.arange(len(tick_center_pixel)):
        img[0:int(tick_height_pixel[ii]),
            int(tick_center_pixel[ii]-0.5*tick_thickness_pixel[ii]):int(tick_center_pixel[ii]+0.5*tick_thickness_pixel[ii]+1)] = 0
    return img    

# create ruler index image
def create_ruler_index_img(img_size_mm=[60.0,2.0], index_xx_mm=5.0, index_depth_mm=2.0, triangle_tip_angle_degrees=45):
    '''Create a triangular index for the scale ruler.
    Input:
        img_size_mm: [mm] size of the image,
        index_xx_mm: [mm] position of the tip of the index.
        index_depth_mm: [mm] yy dimension of the triangular shape.
        triangle_tip_angle_degrees: [degrees] angle of the tip of the triangular shape.
    Output:
        the numpy array of the resulting image.
    '''
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    index_xx_pixel = int(index_xx_mm*img_dpi/25.4)
    yy_flat_surface_triangle_pixel = int((img_size_mm[1]-index_depth_mm)*img_dpi/25.4)
    Tyy = np.arange(img.shape[0]-1, yy_flat_surface_triangle_pixel, -1)
    for ii in np.arange(len(Tyy)):
        triangle_half_width = int( ii*np.tan(np.radians(triangle_tip_angle_degrees/2.0)) )
        img[ Tyy[ii], index_xx_pixel-triangle_half_width:index_xx_pixel+triangle_half_width+1 ] = 0
    return img

# create ruler image
def create_ruler_img(img_size_mm=[200.0,10.0], scale_start_mm=10.0, scale_end_mm=190.00, font_size=0.25*img_dpi, tick_thickness_mm=0.3, text_offset_mm=0.0, number_spacing_mm=10.0, tick_marks_tip_angle_degrees=45):
    '''Create a ruler scale (10 ticks/number).
    Input:
        img_size_mm: [mm] size of the image,
        scale_start_mm: [mm] distance from the left side of the image to the 1st tick,
        scale_end_mm: [mm] distance from the right side of the image to the last tick,
        font_size: [pixels] size of the font,
        tick_thickness_mm: [mm] width of the tick marks, 
        text_offset_mm: [mm] add a vertical offset to the text,
        number_spacing_mm: [mm] spacing between the numbers (eg: use 10.0 for cm, and 25.4 for inches)
        tick_marks_tip_angle_degrees: [degrees] angle of the tip of the tick marks.
    Ouput:
        numpy array for the image.
    '''
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    nb_numbers = 1+int(np.floor((scale_end_mm-scale_start_mm)/number_spacing_mm))
    tick_spacing_mm = number_spacing_mm/10.0
    nb_ticks = 1+np.floor((scale_end_mm-scale_start_mm)/tick_spacing_mm)
    scale_start_pixel = int(scale_start_mm*img_dpi/25.4)
    scale_end_pixel = int(scale_end_mm*img_dpi/25.4)
    tick_center_pixel = np.linspace(scale_start_pixel, scale_end_pixel, nb_ticks).astype(int)
    # ADD THE TEXT
    text_offset_pixel = int(text_offset_mm*img_dpi/25.4)
    font_filename = my_font_filename
    font_size=int(font_size)
    ## Create a temporary PIL image to work on the text
    im = Image.fromarray(img.astype(np.uint8), mode="L") # array filled with 255
    font = ImageFont.truetype(font_filename, font_size)
    draw = ImageDraw.Draw(im)
    ## write the numbers
    for ii in np.arange(nb_numbers):
        draw.text((tick_center_pixel[10*ii]-font.getsize("%d"%ii)[0]/2, int(0.4*img.shape[0]-text_offset_pixel)),"%d"%ii,0,font=font)    
    ## apply the text to the numpy array
    img[np.where(np.asarray(im)==0)] = 0
    # ADD THE TICK MARKS (with pointy heads)
    tick_height_pixel = 0.2*img.shape[0]*np.ones_like(tick_center_pixel)   # the mini ticks
    tick_height_pixel[(5*np.arange(nb_ticks/5.0)).astype(int)] = 0.35*img.shape[0]      # every 5 ticks
    tick_height_pixel[(10*np.arange(nb_ticks/10.0)).astype(int)] = 0.45*img.shape[0]   # every 10 ticks
    tick_thickness_pixel = tick_thickness_mm*img_dpi/25.4 *np.ones_like(tick_height_pixel) # the mini ticks
    tick_thickness_pixel[(5*np.arange(nb_ticks/5.0)).astype(int)] = 1.5* tick_thickness_mm*img_dpi/25.4 # every 5 ticks
    for ii in np.arange(len(tick_center_pixel)):
        for jj in np.arange(int(tick_height_pixel[ii])):
            tick_width = min([ int( jj*np.tan(np.radians(tick_marks_tip_angle_degrees/2.0)) ), tick_thickness_pixel[ii]])
            img[jj, int(tick_center_pixel[ii]-0.5*tick_width):int(tick_center_pixel[ii]+0.5*tick_width+1)] = 0
    return img    


# save image 
def save_img(img, filename='test.png'):
    '''Save an image to disk.
    Input: 
        img: the numpy array for the image,
        filename: the filename.
    '''
    im = Image.fromarray(img.astype(np.uint8), mode="L")
    im.save(filename, dpi=(img_dpi, img_dpi))

    
# stitch images
def stitch_img(img1, img2):
    '''Stitch 2 images, the 1st image on top of the 2nd.
    Input:
        img1: the numpy array for the image on top,
        img2: the numpy array for the image below (must be of the same width).
    Output:
        the numpy array for the resulting image.
    '''
    img = np.zeros((img1.shape[0]+img2.shape[0], img1.shape[1]))
    img[:img1.shape[0],:] = np.copy(img1)
    img[img1.shape[0]:,:] = np.copy(img2)
    return img


# create an image with some text
def create_text_img(img_size_mm=[60.0,10.0], ttext="Hello World!", font_size=0.1*img_dpi, text_xx_center_mm=30.0, text_yy_center_mm=5.0, text_graylevel=0):
    '''Create an image some text on.
    Input:
        img_size_mm: [mm] size of the image,
        ttext: a string of text,
        font_size: [pixels] size of the font,
        text_xx_center_mm: [mm] the center of the text in the xx direction,
        text_yy_center_mm: [mm] the center of the text in the yy direction,
        text_graylevel: [0-255] graylevel of the text.
    Ouput:
        numpy array for the image.
    '''
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    # ADD THE TEXT
    text_xx_center_pixel = int(text_xx_center_mm*img_dpi/25.4)
    text_yy_center_pixel = int(text_yy_center_mm*img_dpi/25.4)
    font_filename = my_font_filename
    font_size=int(font_size)
    ## Create a temporary PIL image to work on the text
    im = Image.fromarray(img.astype(np.uint8), mode="L") # array filled with 255
    font = ImageFont.truetype(font_filename, font_size)
    draw = ImageDraw.Draw(im)
    ## write the text
    draw.text((text_xx_center_pixel-font.getsize(ttext)[0]/2, text_yy_center_pixel-font.getsize(ttext)[1]/2),ttext,0,font=font)    
    ## apply the text to the numpy array
    img[np.where(np.asarray(im)==0)] = text_graylevel
    return img      
    

# create alignement line
def create_alignement_line_img(img_size_mm=[200.0,1.5], line_thickness_mm=0.5, dash_length_mm=2.5, offset_mm=0.0, graylevel=0):
    '''Create a image with a dashed line that can be used as an alignement target.
    Input:
        img_size_mm: [mm] size of the image,
        line_thickness_mm: [mm] thickness of the line,
        dash_length_mm: [mm] length of a dash,
        offset_mm: [mm] offset of the dash,
        graylevel:[0-255] gray level of the line.
    Ouput:
        numpy array for the image.
    '''
    img_size_pixel = [int(img_size_mm[1]*img_dpi/25.4), int(img_size_mm[0]*img_dpi/25.4)]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    line_thickness_pixel = line_thickness_mm*img_dpi/25.4
    dash_length_pixel = dash_length_mm*img_dpi/25.4
    offset_pixel = offset_mm*img_dpi/25.4
    line_yy = np.arange( int(img_size_pixel[0]*0.5-line_thickness_pixel*0.5), int(img_size_pixel[0]*0.5+line_thickness_pixel*0.5))
    for xx in np.arange(img_size_pixel[1]):
        if np.mod(xx-offset_pixel, dash_length_pixel) < (0.5*dash_length_pixel):
            img[line_yy,xx] = graylevel
    return img
    

# add images
def add_images(img01, img02):
    '''Add two images of the same size. The images are assumed to be using only two colors: Black(0) and Transparent(255).
    Input:
        img01: the numpy array for the 1st image,
        img02: the numpy array for the 2nd image,
    Output:
        the numpy array for the resulting image.
    '''
    img = 255*np.ones_like(img01)
    img[np.where(np.logical_or((img01==0),(img02==0)))] = 0
    return img    

# display image
def display_img(img, figureNb=1001):
    '''Display an image.
    Input:
        img: the numpy array for the image,
        figureNb: an ID number the displayed image.
    '''
    plt.figure(figureNb)
    plt.clf()
    plt.imshow(img, cmap='gray')


# add a frame to an image
def add_frame(img_original, frame_thickness_mm = [10,10,10,10], line_thickness_mm=0.5, line_graylevel=0):
    '''Adds a white frame to an image, with a line on the outside of the frame.
    Note: the line is drawn inside the frame thickness (eg: line_thickness does not change the size of the final image).
    Input:
        img_original: the original image,
        frame_thickness_mm: [mm] the thickness of the frame on the [left, right, top, bottom] sides,
        line_thickness_mm: [mm] thickness of the line,
        line_gray_level: [0-255] gray level of the line.
    '''
    img_width_pixel = img_original.shape[1] + int((frame_thickness_mm[0]+frame_thickness_mm[1])*img_dpi/25.4)
    img_height_pixel= img_original.shape[0] + int((frame_thickness_mm[2]+frame_thickness_mm[3])*img_dpi/25.4)
    img_size_pixel = [img_height_pixel, img_width_pixel]  # swap x and y so that Ox is "horizontal" and Oy is "vertical"
    img = 255*np.ones(img_size_pixel)
    # draw the lines
    line_thickness_pixel = int(line_thickness_mm*img_dpi/25.4)
    img[0:line_thickness_pixel, :] = line_graylevel
    img[-line_thickness_pixel:, :] = line_graylevel
    img[:, 0:line_thickness_pixel] = line_graylevel
    img[:, -line_thickness_pixel:] = line_graylevel    
    # place the original image in the frame
    img_00_pixel = [ int(frame_thickness_mm[0]*img_dpi/25.4), int(frame_thickness_mm[2]*img_dpi/25.4) ] # (remember that we swapped Ox and Oy !!!)
    img[ img_00_pixel[1]:(img_00_pixel[1]+img_original.shape[0]), img_00_pixel[0]:(img_00_pixel[0]+img_original.shape[1]) ] = img_original
    return img
    

# Build metric caliper
def build_metric_caliper(filename_base="metric_", measuring_length_mm=100.0):
    '''Build the bottom and top images for a metric caliper.
    Note: the width of the clear aperture for the top frame should be 24.5mm (-ish).
    input:
        filename_base: 1st part of the filename for the images
        measuring_length_mm: [mm] measuring length at full resolution (note: an additional 50mm will be available at coarse resolution)
    return: 
        img_bottom: the numpy array for the saved image for the bottom moire,
        img_top: the numpy array for the saved image for the top moire.
    output:
        saves the two PNG images to disk (bottom and top moires)
    '''
    ## BUILD THE BOTTOM IMAGE
    img_b_niet_1_5mm = 255*np.ones( [int(1.5*img_dpi/25.4),int(70*img_dpi/25.4)] ) # a transparent band
    img_b_dashed_line = create_alignement_line_img([70,1.5], line_thickness_mm=0.25, dash_length_mm=2.5, offset_mm=1.25, graylevel=0)
    img_b_moire_pattern = create_triangular_moire_img(img_size_mm=[70.0,5.0], pitch_mm=1.0*50.0/51.0, line_thickness_mm1=0.6*50.0/51.0, line_thickness_mm2=0.93*50.0/51.0, offset_mm=10.0)
    img_b_moire_scale = create_moire_scale_img(img_size_mm=[70.0,5.0], scale_start_mm=10.0, scale_end_mm=60.0, font_size=0.15*img_dpi, tick_thickness_mm=0.1, text_height_mm=0.0)
    img_b_ruler_index = create_ruler_index_img(img_size_mm=[70.0,2.0], index_xx_mm=10.0, index_depth_mm=2.0, triangle_tip_angle_degrees=45)
    img_b_text = create_text_img(img_size_mm=[70.0,2.0], ttext=u"Moiré Caliper", font_size=0.075*img_dpi, text_xx_center_mm=30.0, text_yy_center_mm=1.0, text_graylevel=0)
    img_b_text2 = create_text_img(img_size_mm=[70.0,2.0], ttext=u"by Mojoptix", font_size=0.075*img_dpi, text_xx_center_mm=55.0, text_yy_center_mm=1.0, text_graylevel=0)
    img_b_ruler_text = add_images(img_b_ruler_index, img_b_text)
    img_b_ruler_text = add_images(img_b_ruler_text, img_b_text2)
    img_b_niet_11_5mm = 255*np.ones( [int(11.5*img_dpi/25.4),int(70*img_dpi/25.4)] ) # a transparent band
    img_bottom = stitch_img(img_b_niet_1_5mm, img_b_dashed_line)
    img_bottom = stitch_img(img_bottom, img_b_moire_pattern)
    img_bottom = stitch_img(img_bottom, img_b_moire_scale)
    img_bottom = stitch_img(img_bottom, img_b_ruler_text)
    img_bottom = stitch_img(img_bottom, img_b_niet_11_5mm)
    # Add the cut-here lines
    img_bottom_final = add_frame(img_bottom, frame_thickness_mm = [15.0,15.0,6.5,6.5], line_thickness_mm=0.25, line_graylevel=50)    
    display_img(img_bottom_final, 101)
    save_img(img_bottom_final, "%sbottom.png"%filename_base)
    ## BUILD THE TOP IMAGE
    full_length_mm = measuring_length_mm+50.0 # add 50.0mm for the bottom image moire 
    full_length_mm = full_length_mm +10.0     # add 5.0mm white space on each side, to have some more room for the moire
    img_t_outside_line01 = create_alignement_line_img([full_length_mm,1.15], line_thickness_mm=1.15,dash_length_mm=1.0, graylevel=0)
    img_t_outside_line02 = create_alignement_line_img([full_length_mm,0.35], line_thickness_mm=0.35,dash_length_mm=10000.0, graylevel=0)
    img_t_dashed_line = create_alignement_line_img([full_length_mm,1.5], line_thickness_mm=0.25,dash_length_mm=2.5,graylevel=100)
    img_t_moire_pattern = create_triangular_moire_img(img_size_mm=[full_length_mm,5.0], pitch_mm=1.0, line_thickness_mm1=0.6, line_thickness_mm2=0.93, offset_mm=0.0)
    img_t_niet_7mm = 255*np.ones( [int(7.0*img_dpi/25.4),int(full_length_mm*img_dpi/25.4)] ) # a transparent band
    img_t_ruler = create_ruler_img(img_size_mm=[full_length_mm,10.0], scale_start_mm=5.0, scale_end_mm=full_length_mm-5.0, font_size=0.25*img_dpi, tick_thickness_mm=0.3, text_offset_mm=0.0, number_spacing_mm=10.0)
    img_top = stitch_img(img_t_outside_line01, img_t_outside_line02)  # alignement target for top image Vs top frame
    img_top = stitch_img(img_top, img_t_dashed_line) # alignement target for bottom image Vs top image
    img_top = stitch_img(img_top, img_t_moire_pattern)
    img_top = stitch_img(img_top, img_t_niet_7mm)
    img_top = stitch_img(img_top, img_t_ruler)
    img_top = stitch_img(img_top, img_t_outside_line02)  # alignement target for top image Vs top frame
    img_top = stitch_img(img_top, img_t_outside_line01)  # alignement target for top image Vs top frame
    # Flip the top image: it will be printed on a transparency that will be used facing down
    img_top_fliplr = np.fliplr(img_top)
    # Add the cut-here lines
    img_top_final = add_frame(img_top_fliplr, frame_thickness_mm = [7.0,7.0,6,6], line_thickness_mm=0.25, line_graylevel=50)
    display_img(img_top_final, 102)
    save_img(img_top_final, "%stop.png"%filename_base)
    # Return
    return (img_bottom_final, img_top_final)    


# Build imperial caliper
def build_imperial_caliper(filename_base="imperial_"):
    img_imperial = create_text_img(img_size_mm=[200,50], ttext="I am afraid I can't do that, Dave.", font_size=0.5*img_dpi, text_xx_center_mm=100, text_yy_center_mm=25, text_graylevel=0)
    save_img(img_imperial, "%simage.png"%filename_base)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Build the images for a Metric Caliper and display the moires obtained at 0.0 mm and at 0.5mm
# -----------------------------------------------------------------------------
(img_bottom, img_top) = build_metric_caliper();
## Display the Caliper at 0.000mm
img_test_top = np.fliplr(img_top)
yy_offset_pixel = int(0.5*img_dpi/25.4)
xx_offset_pixel = int(5.0*img_dpi/25.4)
img_test = add_images(img_bottom[yy_offset_pixel:(yy_offset_pixel+img_top.shape[0]),xx_offset_pixel:], img_test_top[:,0:(img_bottom.shape[1]-xx_offset_pixel)])
display_img(img_test, 103)
## Display the Caliper at 0.500mm
xx_offset_pixel = int(10.5*img_dpi/25.4)
img_test = add_images(img_bottom[yy_offset_pixel:(yy_offset_pixel+img_top.shape[0]),xx_offset_pixel:], img_test_top[:,0:(img_bottom.shape[1]-xx_offset_pixel)])
display_img(img_test, 104)    


    