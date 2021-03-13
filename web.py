import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
from PIL import Image, ImageOps
import cv2
import numpy as np

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	img = np.array(img)
	return img 

def mask_generate(img,dark,light):
  dark_black = dark
  light_black = light
  mask = cv2.inRange(img,dark_black,light_black)
  area = cv2.bitwise_and(img,img,mask=mask)
  area = cv2.medianBlur(area,5)
  return area

def img_thresh(crop):
  crop = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
  _, threshold = cv2.threshold(crop,0,255,cv2.THRESH_OTSU)
  return threshold

def bright_img(img):
  img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
  blur = cv2.blur(img, (5, 5))  # With kernel size depending upon image size
  if cv2.mean(blur)[2] > 160:  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
      return 'light' # (127 - 255) denotes light image
  else:
      return 'dark' # (0 - 127) denotes dark image

def ccontours(img):

  img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours.sort (key = lambda x: cv2.contourArea (x), reverse = True)

  h, w = img.shape[:2]
  mask = np.zeros((h, w), np.uint8)

  cv2.drawContours(mask, [contours[0]],-1, 255, -1)
 
  num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(mask,8,cv2.CV_32S)

  stats = stats[1:,]
  left = min(stats[:,0])-50
  if left < 0 :
    left = 0
  top = min(stats[:,1])-50
  if top < 0 :
    top = 0
  right = min(stats[:,0]) + max(stats[:,2])+50
  if img.shape[1]<right:
    right = img.shape[1]
  bottom = min(stats[:,1]) + max(stats[:,3])+50
  if img.shape[0] < bottom:
    bottom = img.shape[0]

  crop = img.copy()
  crop = crop[top:bottom,left:right,:]
  
  return mask,crop

def lightpre(img):
  low_black = (0,0,0)
  high_black = (135,135,135)
  mask = mask_generate(img,low_black,high_black)
  m,contours = ccontours(mask)
  threshold = img_thresh(contours)
  result = cv2.resize(threshold,(224,224), interpolation = cv2.INTER_AREA)
  return result

def darkpre(img):
  gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
  gray_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
          cv2.THRESH_BINARY,11,2)  
  gray_img =  cv2.fastNlMeansDenoising(gray_img,None,10,7,21)
  kernel = np.ones((2,2),np.uint8)
  dilate = cv2.dilate(gray_img,kernel,iterations=2)
  gray_img = cv2.bitwise_not(dilate)
  res = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
  m,contours = ccontours(res)
  result = cv2.resize(contours,(224,224), interpolation = cv2.INTER_AREA)
  return result

def main():
	st.title("Test Input")

	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		first_test = st.radio("Both hands strecth out to front and all fingers strecth out as big as they can for 30 seconds. Did your hands shake ? ", ("Yes", "No"))

		if first_test == 'Yes':
			result_first = 1	
		elif first_test == 'No':
			result_first= 0

		second_test = st.radio("Try to drink water from a cup. Did your hands shake ? ", ("Yes", "No"))
		if second_test == 'Yes':
			result_second = 1
		elif second_test == 'No':
			result_second = 0

		third_test = st.radio("Put both hands on each hips and each hand pose like the alphabet 'C'. Did your hands shake ? ", ("Yes", "No"))

		if third_test == 'Yes':
			result_third = 1
		elif third_test == 'No':
			result_third = 0
			
		example_spiral = Image.open("Spiral.png")
		st.write("Using a black pen and  a white, clean paper, draw a spiral like below without your drawing hand touching the paper")
		st.image(example_spiral) 
		st.write("With your phone camera and the Flash On, Take a photo of your result")
		image_file = st.file_uploader("Upload Your Spiral Result",type=['png','jpeg','jpg'])
		if image_file is not None:
		
			# file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
			# img = cv2.imdecode(file_bytes, 1)
			# st.image(img, channels="BGR")
			img = load_image(image_file)
			img = cv2.medianBlur(img,5)
			status = bright_img(img)
			if status == "light" :
				res = lightpre(img)
				st.write("light")
				st.write(res.shape)
				st.image(res)
			if status == "dark" :
				res = darkpre(img)
				st.write("dark")
				st.write(res.shape)
				st.image(res)

		if st.button("Done"):
			st.write( "[" + str(result_first) + " , "+ str(result_second) +" , " + str(result_third) + "]")
			st.write("The result is Essential Tremor")			
	else:
		st.subheader("About")
		st.info("Built with Streamlit")
		st.info("Jesus Saves @JCharisTech")
		st.text("Jesse E.Agbe(JCharis)")

if __name__ == '__main__':
	main()
