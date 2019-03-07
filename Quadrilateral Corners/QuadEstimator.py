import numpy as np
import cv2


class QuadEstimator():
	
	def __init__(self, params = None):
		pass
		
		if params is None:
			self.params = {
				'MIN_SHAPE_AREA' : 250,
				'EPSILON_K_FIRST' : 0.036,
				'EPSILON_K_SECOND' : 0.05,
				'DRAW_DEBUG_IMAGES' : True
			}

		
	def _draw_points_array(self, image, points, color=(255,0,0), size=7):
		if points is not None: 
			for i in range(0, len(points)):
				cv2.circle(image, (int(points[i,0]), int(points[i,1])), 10, color, size)
				
		return image

	
	def _preprocess_img(self, img_original, gray=False, threshold_bw=50, blur=5):

		if gray is False:
			img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
		else:
			img_gray = img_original

		img_blur =  cv2.GaussianBlur(img_gray, (blur, blur), 0)
		img_bw = cv2.threshold(img_blur,threshold_bw,255,cv2.THRESH_BINARY)[1]
		
		return img_bw
	

	### Do I need???
	def find_and_draw_contours(self, img_path):
		
		img_original = cv2.imread(img_path)
		
		img_bw = self.preprocess_img(img_original)
		
		cnts = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		
		for index,c in enumerate(cnts):
			cv2.drawContours(img_original, [c], -1, (0,255,0), 3)
		return img_bw

	
	def _find_corners_from_approx(self, img_bw, approx):
		# create mask for edge detection
		gray = np.float32(img_bw)
		mask = np.zeros(gray.shape, dtype="uint8")
		cv2.fillPoly(mask, [approx], (255,255,255))

		dst = cv2.cornerHarris(mask,5,3,0.04)
		ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
		dst = np.uint8(dst)
		ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
		
		return corners
	
	def _find_shapes(self, img_bw, img_original, debug=True):
		

		img_result = img_original.copy()

		results = []

		#cnts = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		_, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	   
		if debug is True:
			print('Contours:', len(contours))
			cv2.drawContours(img_result, contours, -1, (0,255,0), 3)

		#cnts = imutils.grab_contours(cnts)
				
		for index, c in enumerate(contours):


			epsilon = self.params['EPSILON_K_FIRST'] * cv2.arcLength(c,True)
			approx = cv2.approxPolyDP(c, epsilon, True)

			if debug:
				cv2.drawContours(img_result, approx, -1, (255, 0, 0), 3)
				print('* Shape', index,'countours',len(c),'with',len(approx),'approx')
			

			# 4 sides?
			if (len(approx)>2):
				
				cv2.drawContours(img_result, [c], -1, (0, 255, 0), 5)

				#(x, y, w, h) = cv2.boundingRect(approx)
				#cv2.rectangle(img_result,(x,y),(x+w,y+h),(0,0,255),3)
				#print('Shape:',index,'x, y, w, h, corners:',x, y, w, h, len(approx))
				
				area = cv2.contourArea(approx)
				
				
				
				
				if (area>self.params['MIN_SHAPE_AREA']):
					
					print('- Area:',area)
					
					result = {}
					result['area'] = area


					# Create mask from shape
					mask = np.zeros(img_bw.shape, dtype="uint8")
					cv2.fillPoly(mask, [approx], (255,255,255))
					#plt.imshow(mask)
					#plt.title('Mask')
					#plt.show()


					### --- Harris corners
					dst = cv2.cornerHarris(mask,5,3,0.02) # 5, 3, 0.04
					ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
					dst = np.uint8(dst)
					ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
					criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
					corners = cv2.cornerSubPix(mask, np.float32(centroids),(5,5),(-1,-1),criteria) # (5,5)
					# first element is a centroid we dont need
					corners = corners[1:]

					print('Harris corners:',len(corners), 'approx corners:', len(approx))

					# Handling weird cases
					if len(corners)!=4 and len(approx)==4:
						print('Harris failed! using approx instead')
						corners = np.reshape(approx, (len(approx),2))
					
					## --- Corners result logic
					if len(corners)==4:

						print('--- First pass solution')
						print(len(corners))
						# print(corners)

						# Draw circles
						for i in range(0, len(corners)):
							#print(corners[i,0])
							cv2.circle(img_result, (int(corners[i,0]), int(corners[i,1])), 10, (255,0,0), 5)

						result['solution_pass'] = 1
						result['corners'] = corners

						results.append(result)

					elif len(corners)>4:


						# Try hull convex approach


						hull = cv2.convexHull(approx)
						#print(hull)
						mask_output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
						cv2.drawContours(img_result, [hull], -1, (255,0,0), 3)
						
						print('hull corners: ', len(hull))

						if len(hull)!=4:
							print('--- Third pass solution')
							# Try second pass approx
							epsilon = self.params['EPSILON_K_SECOND'] * cv2.arcLength(c,True) #0.085
							approx = cv2.approxPolyDP(hull,epsilon,True)

							cv2.drawContours(img_result, [approx], -1, (0,255,0), 3)

							approx = np.reshape(approx, (len(approx),2))

							print('corners', len(approx))

							result['solution_pass'] = 3
							result['corners'] = approx
							if len(approx)==4:
								results.append(result)

						else:
							print('--- Second pass solution')

							approx = np.reshape(hull, (len(hull),2))

							print(len(approx))

							result['solution_pass'] = 2
							result['corners'] = approx
							if len(approx)==4:
								results.append(result)

						# Draw circles
						for i in range(0, len(approx)):
							#print(corners[i,0])
							cv2.circle(img_result, (int(approx[i][0]), int(approx[i][1])), 10, (255,0,0), 5)
						
						

					else: # less than 4 corners
						# Try second pass approx
						print('--- Second pass with 3 corners')						
						hull = cv2.convexHull(c)
						
						cv2.drawContours(img_result, [hull], -1, (255,0,0), 3)

						epsilon = 0.015 * cv2.arcLength(c,True) #0.085
						approx = cv2.approxPolyDP(hull,epsilon,True)
						print('3 corners second pass:',len(approx))
						cv2.drawContours(img_result, [approx], -1, (0,0,255), 3)
						approx = np.reshape(approx, (len(approx),2))
						
						if len(approx)==4:
							result['solution_pass'] = 3
							result['corners'] = approx

							results.append(result)
						
					

		return img_result, results
	


	def _get_inner_area_corners_from_results(self, results):
		print(results)
		try: 
			assert len(results)<3        
		except:
			print('** I got more than 2 shapes for a gate...', len(results))
			
		try: 
			assert len(results)>0
		except: 
			print("** I got no shapes!")
			return None
			
		if len(results) == 1:
			print('** One candidate selected.')
			inner_shape = results[0]
		else: # TODO: Address more than 2 condidates
			print('** Two candidates. Selecting the bigger area')
			inner_shape = None
			if results[0]['area'] > results[1]['area'] and len(results[0]['corners'])==4:
				inner_shape = results[1]
			else:
				inner_shape = results[0]
				
		if inner_shape is not None and 'corners' in inner_shape and len(inner_shape['corners'])==4:
			return inner_shape['corners']
		else:
			return None

	def process_img(self, img_original, gray=False):
		img_bw = self._preprocess_img(img_original, gray)
		
		img_shapes, result = self._find_shapes(img_bw, img_original)
		
		corners = self._get_inner_area_corners_from_results(result)
		
		return corners, img_shapes
	
	def process_img_path(self, img_path):
		img_original = cv2.imread(img_path)
		
		corners, img_shapes = self.process_img(img_original)

		if corners is not None and len(corners)>0:
			img_solution = self._draw_points_array(img_original.copy(), corners)
		else:
			print('** No solution for:',img_path)
			img_solution = img_shapes
		
		return corners, img_solution

