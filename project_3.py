
import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow
from tabulate import tabulate

#Importing images of Canvas and Labeled

campus = cv.imread('/content/Campus.jpg')
labeled = cv.imread('/content/Labeled.pgm')
newImg = labeled.copy()
table = open("Table.txt", "r")

campus = cv.cvtColor(campus, cv.COLOR_BGR2GRAY)
labeled = cv.cvtColor(labeled, cv.COLOR_BGR2GRAY)

#Threshold here chosen to be low enough to recognize the darkest building (Pupin)
ret, thresh = cv.threshold(labeled, 8, 255, 0)


contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#By drawing the different contours, I found that 24 is the empty space of Mud&Co, so
#I deleted contour number 24 from the list of contours
contours = np.delete(contours, [24])
#cv.drawContours(newImg, contours, 23, (0,255,0), 3)
#cv2_imshow(newImg)

#Since the contours start from the largest pixels (Carmen to Pupin), I changed the
#contours array to go backwards to make more intuitive sense
contours = contours[::-1]
#cv.drawContours(newImg, contours, 0, (0,255,0), 3)
#cv2_imshow(newImg)

areas = []

#finding area of each contour
for cnt in contours:
  area = cv.contourArea(cnt)
  areas.append(area)


pixels = []
buildings = []

#creating a list of buildings by number/pixel number
for i in range(labeled.shape[0]):
    for j in range (labeled.shape[1]):
        pixels.append(labeled[i][j])
        if labeled[i][j] > 0:
          if labeled[i][j] not in buildings:
            buildings.append(labeled[i][j])


#created this for a sanity check to makes sure my areas were correct
real_areas = []
for x in buildings:
  pixels_area = 0
  for i in range(labeled.shape[0]):
    for j in range(labeled.shape[1]):
      if labeled[i][j] == x:
        pixels_area += 1
  real_areas.append(pixels_area)


xycoordinates = []
horizontality = []
verticality = []
approxes = []

for cnt in contours:
  approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
  approxes.append(approx)
  M1 = cv.moments(approx)
  cX = int(M1["m10"] / M1["m00"])
  cY = int(M1["m01"] / M1["m00"])
  xycoordinates.append((cX, cY))
  horizontality.append(cX)
  verticality.append(cY)

upperleftcorners = []
lowerrightcorners = []
diagonals = []

heights = []
widths = []

for cnt in contours:
  x,y,w,h = cv.boundingRect(cnt)
  upperleftcorners.append((x,y))
  lowerrightcorners.append((x+w, y+h))
  heights.append(h)
  widths.append(w)
  diagonal = np.sqrt(h**2 + w**2)
  diagonals.append(diagonal)

for i in range(26):
  cv.rectangle(newImg,upperleftcorners[i], lowerrightcorners[i],(0,255,0),2)



buildings_dict = {'9': 'Pupin', '19': 'SchapiroCEPSR', '28': 'Mudd&EngTerrace&Fairchild&CS',
                  '38': 'NorthwestCorner', '47': 'Uris', '57': 'Schermerhorn', '66' : 'Chandler&Havemeyer',
                  '76': 'OldComputerCenter', '85': 'Avery', '94': 'Fayerweather', '104': 'Mathematics',
                  '113': 'LowLibrary', '123': 'StPaulChapel', '132': 'EarlHall', '142': 'Lewisohn',
                  '151': 'Philosophy', '161': 'Buell', '170': 'AlmaMater', '179': 'Dodge', '189': 'Kent',
                  '198': 'CollegeWalk', '208': 'Journalism&Furnald', '217': 'Hamilton&Hartley&Wallach&JohnJay',
                  '236': 'Lerner', '246': 'ButlerLibrary', '255': 'Carman'}

intersecting_dict = {'9': 'None', '19': 'None', '28': 'Uris, Schermerhorn',
                  '38': 'None', '47': 'Mudd&EngTerrace&Fairchild&CS', '57': 'Mudd&EngTerrace&Fairchild&CS', '66' : 'None',
                  '76': 'None', '85': 'None', '94': 'None', '104': 'None',
                  '113': 'None', '123': 'None', '132': 'None', '142': 'None',
                  '151': 'None', '161': 'None', '170': 'None', '179': 'None', '189': 'None',
                  '198': 'None', '208': 'None', '217': 'None',
                  '236': 'None', '246': 'None', '255': 'None'}

for i in buildings:
  exec('tabletake%s = [["Building Number: %s", "Values:"], ["Building Name: ", 0], ["(x,y) Centre: ", 0], ["Area in Pixels: ", 0], \
  ["Upper Left MBR: ", 0],["Lower Right MBR: ", 0], ["MBR Diagonal: ", 0], ["Other Buildings: ", "None"]]' % (i, i))


for i in buildings:
  exec('tabletake%s[1][1] = buildings_dict.get("%s")' %(i, i))
  exec('tabletake%s[7][1] = intersecting_dict.get("%s")' %(i, i))


for i in range(26):
  exec('tabletake%s[3][1] = %s' %(buildings[i], areas[i]))
  exec('tabletake%s[2][1] = %s' %(buildings[i], xycoordinates[i]))
  exec('tabletake%s[4][1] = %s' %(buildings[i], upperleftcorners[i]))
  exec('tabletake%s[5][1] = %s' %(buildings[i], lowerrightcorners[i]))
  exec('tabletake%s[6][1] = %s' %(buildings[i], diagonals[i]))


for i in buildings:
  exec('print(tabulate(tabletake%s, headers="firstrow"))' % (i))
  print('\n')


print(horizontality)

train_apple1 = cv.imread('/content/train_apple1.jpg', cv.IMREAD_COLOR)
train_apple2 = cv.imread('/content/train_apple2.jpg', cv.IMREAD_COLOR)
train_apple3 = cv.imread('/content/train_apple3.jpg', cv.IMREAD_COLOR)
train_apple4 = cv.imread('/content/train_apple4.jpg', cv.IMREAD_COLOR)
train_apple5 = cv.imread('/content/train_apple5.jpg', cv.IMREAD_COLOR)

train_orange1 = cv.imread('/content/train_orange1.jpg', cv.IMREAD_COLOR)
train_orange2 = cv.imread('/content/train_orange2.jpg', cv.IMREAD_COLOR)
train_orange3 = cv.imread('/content/train_orange3.jpg', cv.IMREAD_COLOR)
train_orange4 = cv.imread('/content/train_orange4.jpg', cv.IMREAD_COLOR)
train_orange5 = cv.imread('/content/train_orange5.jpg', cv.IMREAD_COLOR)

train_banana1 = cv.imread('/content/train_banana1.jpg', cv.IMREAD_COLOR)
train_banana2 = cv.imread('/content/train_banana2.jpg', cv.IMREAD_COLOR)
train_banana3 = cv.imread('/content/train_banana3.jpg', cv.IMREAD_COLOR)
train_banana4 = cv.imread('/content/train_banana4.jpg', cv.IMREAD_COLOR)
train_banana5 = cv.imread('/content/train_banana5.jpg', cv.IMREAD_COLOR)

areas.sort()
#print(areas)

aspects = []

for i in range(26):
  aspect_ratio = heights[i] / widths[i]
  aspects.append(aspect_ratio)

#for i in range(26):
  #print(len(approxes[i]))


buildings_size = ['Mid-Size', 'Mid-Size', 'Largest', 'Mid-Size', 'Largest', 'Large', 'Large', 'Smallest',
                  'Small', 'Small', 'Small', 'Large', 'Small', 'Small', 'Mid-Size', 'Small', 'Smallest',
                  'Smallest', 'Mid-Size', 'Mid-Size', 'Largest', 'Large', 'Largest', 'Large', 'Largest',
                  'Mid-Size']
buildings_aspects = ['Wide', 'Medium', 'Medium', 'Narrow', 'Narrow', 'Medium', 'Medium', 'Narrow', 'Narrow',
                     'Narrow', 'Narrow', 'Medium', 'Wide', 'Medium', 'Narrow', 'Narrow', 'Medium', 'Medium',
                     'Wide', 'Wide', 'Wide', 'Medium', 'Narrow', 'Wide', 'Medium', 'Wide']
buildings_geometry = ['Rectangle', 'Rectangle', 'Asymmetrical', 'Rectangle', 'Rectangle', 'L', 'L', 'Rectangle', 'C', 'Rectangle', 'Rectangle',
                      'Square', 'Rectangle', 'I', 'I', 'I', 'I', 'Square', 'C', 'C', 'Rectangle',
                      'Asymmetrical', 'Asymmetrical', 'Rectangle', 'I', 'Rectangle']
building_confusion = ['Carman', 'None', 'None', 'None', 'None', 'Chandler&Havemeyer', 'Schermerhorn', 'None', 'None', 'Mathematics', 'Fayerweather',
                      'None', 'None', 'None', 'None', 'None', 'None', 'None', 'Kent', 'Dodge', 'None', 'None',
                      'None', 'None', 'None', 'Pupin']

building_minimization = [['Mid-Size', 'Wide', 'Rectangle'], ['Mid-Size', 'Medium'], ['Largest', 'Medium', 'Asymmetrical'],
                         ['Mid-Size', 'Narrow', 'Rectangle'], ['Largest', 'Narrow', 'Rectangle'], ['Large', 'Medium', 'L'], ['Large', 'Medium', 'L'],
                         ['Smallest', 'Narrow'], ['Narrow', 'C'], ['Small', 'Narrow', 'Rectangle'], ['Small', 'Narrow', 'Rectangle'],
                        ['Large', 'Medium', 'Square'], ['Small', 'Wide'], ['Small', 'Medium'], ['Mid-Size', 'Narrow', 'I'], ['Small', 'Narrow', 'I'],
                         ['Smallest', 'Medium', 'I'], ['Smallest', 'Medium', 'Square'], ['Mid-Size', 'Wide', 'C'], ['Mid-Size', 'Wide', 'C'],
                         ['Largest', 'Wide'], ['Large', 'Medium', 'Asymmetrical'], ['Narrow', 'Asymmetrical'], ['Large', 'Wide'], ['Largest', 'Medium', 'I'],
                         ['Mid-Size', 'Wide', 'Rectangle']]


for i in buildings:
  exec('tabletake%s = [["Building Number: %s", "Values:"], ["Building Name: ", 0], ["Size: ", "Medium"], ["Aspect Ratio: ", 0], \
  ["Geometry: ", 0], ["Confusion: ", 0], ["Minimization:", 0]]' % (i, i))

for i in buildings:
  exec('tabletake%s[1][1] = buildings_dict.get("%s")' %(i, i))

for i in range(26):
  exec('tabletake%s[2][1] = "%s"' %(buildings[i], buildings_size[i]))
  exec('tabletake%s[3][1] = "%s"' %(buildings[i], buildings_aspects[i]))
  exec('tabletake%s[4][1] = "%s"' %(buildings[i], buildings_geometry[i]))
  exec('tabletake%s[5][1] = "%s"' %(buildings[i], building_confusion[i]))
  exec('tabletake%s[6][1] = "%s"' %(buildings[i], building_minimization[i]))

for i in buildings:
  exec('print(tabulate(tabletake%s, headers="firstrow"))' % (i))
  print('\n')

verticality.sort()
horizontality.sort()

#print(verticality)
#print(horizontality)


verticalhorizontal = np.subtract(heights, widths)
verticalhorizontal.sort()
#print(verticalhorizontal)

buildings_verticality = ['Uppermost', 'Uppermost', 'Uppermost', 'Uppermost', 'Uppermost', 'Upper',
                        'Upper', 'Upper', 'Upper', 'Upper', 'Mid-height', 'Mid-height', 'Mid-height',
                         'Mid-height', 'Mid-height', 'Mid-height', 'Lower', 'Lower', 'Lower', 'Lower',
                         'Lower', 'Lowermost', 'Lowermost', 'Lowermost', 'Lowermost', 'Lowermost',]
buildings_horizontality = ['Left', 'Mid-width', 'Right', 'Left-most', 'Mid-width', 'Right-most',
                           'Left-most', 'Mid-width', 'Right', 'Right-most', 'Left-most', 'Mid-width',
                           'Right', 'Left', 'Left-most', 'Right-most', 'Right', 'Mid-width', 'Left',
                           'Right-most', 'Mid-width', 'Left-most', 'Right-most', 'Left', 'Mid-width', 'Left']

buildings_orientation = ['Horizontal', 'Non-oriented', 'Horizontal', 'Vertical', 'Vertical', 'Horizontal',
                         'Non-oriented', 'Non-oriented', 'Vertical', 'Vertical', 'Vertical', 'Non-oriented',
                         'Horizontal', 'Non-oriented', 'Vertical', 'Vertical', 'Non-oriented', 'Non-oriented',
                         'Horizontal', 'Horizontal', 'Horizontal', 'Non-oriented', 'Vertical', 'Horizontal',
                         'Horizontal', 'Horizontal']
buildings_confusion = ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'Lewisohn',
                       'None', 'None', 'None', 'Mathematics', 'None', 'None', 'None', 'None', 'None', 'None', 'None',
                       'None', 'Carmen', 'None', 'Lerner']
buildings_minimization = [['Uppermost', 'Left'], ['Uppermost', 'Mid-width', 'Non-oriented'], ['Uppermost', 'Right'],
                          ['Uppermost', 'Left-most'], ['Uppermost', 'Mid-width', 'Vertical'], ['Upper', 'Right-most', 'Horizontal'],
                          ['Upper', 'Left-most'], ['Upper', 'Mid-width'], ['Upper', 'Right'], ['Upper', 'Right-most', 'Vertical'],
                          ['Mid-height', 'Left-most', 'Vertical'], ['Mid-height', 'Mid-width'], ['Mid-height', 'Right'],
                          ['Mid-height', 'Left'], ['Mid-height', 'Left-most', 'Vertical'], ['Mid-height', 'Right-most'], ['Lower', 'Right'],
                          ['Lower', 'Mid-width', 'Non-oriented'], ['Lower', 'Left'], ['Lower', 'Right-most'], ['Lower', 'Mid-width', 'Horizontal'],
                          ['Lowermost', 'Left-most'], ['Lowermost', 'Right-most'], ['Lowermost', 'Left', 'Horizontal'], ['Lowermost', 'Mid-width'],
                          ['Lowermost', 'Left', 'Horizontal']]

for i in buildings:
  exec('tabletake%s = [["Building Number: %s", "Values:"], ["Building Name: ", 0], ["Verticality: ", "left"], ["Horizontality: ", 0], \
  ["Orientation: ", 0],["Confusion: ", 0], ["Minimization: ", 0]]' % (i, i))

print(len(buildings_minimization))

for i in buildings:
  exec('tabletake%s[1][1] = buildings_dict.get("%s")' %(i, i))

for i in range(26):
  exec('tabletake%s[2][1] = "%s"' %(buildings[i], buildings_verticality[i]))
  exec('tabletake%s[3][1] = "%s"' %(buildings[i], buildings_horizontality[i]))
  exec('tabletake%s[4][1] = "%s"' %(buildings[i], buildings_orientation[i]))
  exec('tabletake%s[5][1] = "%s"' %(buildings[i], buildings_confusion[i]))
  exec('tabletake%s[6][1] = "%s"' %(buildings[i], buildings_minimization[i]))




for i in buildings:
  exec('print(tabulate(tabletake%s, headers="firstrow"))' % (i))
  print('\n')

print(xycoordinates)


def nearTo(S, T):

  is_near = False

  if abs(S[0] - T[0]) < 90:
    if abs(S[1] - T[1]) < 90:
      is_near = True

  return is_near


print(nearTo((223, 34),(233, 120)))

##Testing nearTo function###


source = (38, 479)
print("Pupin", '\t','\t','\t','\t', nearTo(source,(76, 14)))
print("SchapiroCEPSR", '\t','\t','\t', nearTo(source,(143, 20)))
print("Mudd&EngTerrace&Fairchild&CS", '\t', nearTo(source,(223, 34)))
print("NorthwestCorner", '\t','\t', nearTo(source,(16, 40)))
print("Uris", '\t','\t','\t','\t', nearTo(source,(142, 99)))
print("Schermerhorn", '\t', '\t', '\t', nearTo(source,(233, 120)))
print("Chandler&Havemeyer", '\t', '\t', nearTo(source,(37, 120)))
print("OldComputerCenter", '\t', '\t', nearTo(source,(96, 136)))
print("Avery", '\t', '\t', '\t', '\t', nearTo(source,(204, 175)))
print("Fayerweather", '\t', '\t', '\t', nearTo(source,(259, 176)))
print("Mathematics", '\t', '\t', '\t', nearTo(source,(17, 182)))
print("LowLibrary", '\t', '\t', '\t', nearTo(source,(135, 221)))
print("StPaulChapel", '\t', '\t', '\t', nearTo(source,(226, 222)))
print("EarlHall", '\t', '\t', '\t', nearTo(source,(49, 221)))
print("Lewisohn", '\t', '\t', '\t', nearTo(source,(17, 259)))
print("Philosophy", '\t', '\t', '\t',nearTo(source,(258, 263)))
print("Buell", '\t','\t', '\t', '\t',nearTo(source,(208, 253)))
print("AlmaMater", '\t', '\t', '\t',nearTo(source,(136, 276)))
print("Dodge", '\t', '\t', '\t', '\t',nearTo(source,(41, 301)))
print("Kent", '\t', '\t', '\t', '\t',nearTo(source,(233, 300)))
print("CollegeWalk", '\t', '\t', '\t',nearTo(source,(137, 322)))
print("Journalism&Furnald	", '\t',nearTo(source,(30, 363)))
print("Hamilton&Hartley&Wallach&JohnJay", nearTo(source,(240, 417)))
print("Lerner", '\t', '\t', '\t', '\t', nearTo(source,(38, 446)))
print("ButlerLibrary", '\t', '\t', '\t', nearTo(source,(132, 460)))
print("Carman", '\t', '\t', '\t', '\t',nearTo(source,(38, 479)))

target = (259, 176)
print("Pupin", '\t','\t','\t','\t', nearTo((76, 14), target))
print("SchapiroCEPSR", '\t','\t','\t', nearTo((143, 20), target))
print("Mudd&EngTerrace&Fairchild&CS", '\t', nearTo((223, 34), target))
print("NorthwestCorner", '\t','\t', nearTo((16, 40), target))
print("Uris", '\t','\t','\t','\t', nearTo((142, 99), target))
print("Schermerhorn", '\t', '\t', '\t', nearTo((233, 120), target))
print("Chandler&Havemeyer", '\t', '\t', nearTo((37, 120), target))
print("OldComputerCenter", '\t', '\t', nearTo((96, 136), target))
print("Avery", '\t', '\t', '\t', '\t', nearTo((204, 175), target))
print("Fayerweather", '\t', '\t', '\t', nearTo((259, 176), target))
print("Mathematics", '\t', '\t', '\t', nearTo((17, 182), target))
print("LowLibrary", '\t', '\t', '\t', nearTo((135, 221), target))
print("StPaulChapel", '\t', '\t', '\t', nearTo((226, 222), target))
print("EarlHall", '\t', '\t', '\t', nearTo((49, 221), target))
print("Lewisohn", '\t', '\t', '\t', nearTo((17, 259), target))
print("Philosophy", '\t', '\t', '\t',nearTo((258, 263), target))
print("Buell", '\t','\t', '\t', '\t',nearTo((208, 253), target))
print("AlmaMater", '\t', '\t', '\t',nearTo((136, 276), target))
print("Dodge", '\t', '\t', '\t', '\t',nearTo((41, 301), target))
print("Kent", '\t', '\t', '\t', '\t',nearTo((233, 300), target))
print("CollegeWalk", '\t', '\t', '\t',nearTo((137, 322), target))
print("Journalism&Furnald	", '\t',nearTo((30, 363), target))
print("Hamilton&Hartley&Wallach&JohnJay", nearTo((240, 417), target))
print("Lerner", '\t', '\t', '\t', '\t', nearTo((38, 446), target))
print("ButlerLibrary", '\t', '\t', '\t', nearTo((132, 460), target))
print("Carman", '\t', '\t', '\t', '\t',nearTo((38, 479), target))

nearness_target = [['SchapiroCEPSR', 'NorthwestCorner', 'Uris'], ['Pupin', 'Mudd&EngTerrace&Fairchild&CS', 'Uris'], ['SchapiroCEPSR', 'Uris', 'Schermerhorn'],
                   ['Pupin', 'Chandler&Havemeyer'], ['Pupin','SchapiroCEPSR', 'Mudd&EngTerrace&Fairchild&CS', 'OldComputerCenter', 'Avery'],
                   ['Mudd&EngTerrace&Fairchild&CS','Avery', 'Fayerweather'], ['NorthwestCorner', 'OldComputerCenter', 'Mathematics'], ['Chandler&Havemeyer ', 'Uris', 'Mathematics', 'EarlHall', 'LowLibrary'],
                   ['Schermerhorn', 'Uris', 'Fayerweather', 'LowLibrary', 'Philosophy', 'Buell'], ['Schermerhorn', 'Avery', 'StPaulChapel', 'Philosophy', 'Buell'], ['Chandler&Havemeyer', 'OldComputerCenter', 'EarlHall', 'Lewisohn'],
                   ['OldComputerCenter', 'Avery', 'Buell', 'EarlHall', 'AlmaMater'], ['Avery', 'Fayerweather', 'Philosophy', 'Buell', 'Kent'], ['OldComputerCenter', 'Mathematics', 'LowLibrary', 'Lewisohn', 'AlmaMater', 'Dodge'],
                   ['Mathematics', 'EarlHall', 'Dodge'], ['Avery', 'Fayerweather', 'StPaulChapel', 'Buell', 'Kent'], ['Avery', 'Fayerweather', 'LowLibrary', 'StPaulChapel', 'Philosophy', 'AlmaMater', 'Kent', 'CollegeWalk'],
                   ['LowLibrary ', 'EarlHall', 'Buell', 'CollegeWalk'], ['EarlHall', 'Lewisohn', 'Journalism&Furnald'], ['StPaulChapel ', 'Philosophy', 'Buell', ], ['Buell', 'AlmaMater'], ['Dodge', 'Lerner'],
                   ['None',], ['Journalism&Furnald', 'Carman'], ['None',], ['Lerner',]]


nearness_source = [['SchapiroCEPSR', 'NorthwestCorner', 'Uris'], ['Pupin', 'Mudd&EngTerrace&Fairchild&CS', 'Uris'], ['SchapiroCEPSR', 'Uris', 'Schermerhorn'],
                   ['Pupin', 'Chandler&Havemeyer'], ['Pupin','SchapiroCEPSR', 'Mudd&EngTerrace&Fairchild&CS', 'OldComputerCenter', 'Avery'],
                   ['Mudd&EngTerrace&Fairchild&CS','Avery', 'Fayerweather'], ['NorthwestCorner', 'OldComputerCenter', 'Mathematics'], ['Chandler&Havemeyer ', 'Uris', 'Mathematics', 'EarlHall', 'LowLibrary'],
                   ['Schermerhorn', 'Uris', 'Fayerweather', 'LowLibrary', 'Philosophy', 'Buell'], ['Schermerhorn', 'Avery', 'StPaulChapel', 'Philosophy', 'Buell'], ['Chandler&Havemeyer', 'OldComputerCenter', 'EarlHall', 'Lewisohn'],
                   ['OldComputerCenter', 'Avery', 'Buell', 'EarlHall' 'AlmaMater'], ['Avery', 'Fayerweather', 'Philosophy', 'Buell', 'Kent'], ['OldComputerCenter', 'Mathematics', 'LowLibrary', 'Lewisohn', 'AlmaMater', 'Dodge'],
                   ['Mathematics', 'EarlHall', 'Dodge'], ['Avery', 'Fayerweather', 'StPaulChapel', 'Buell', 'Kent'], ['Avery', 'Fayerweather', 'LowLibrary', 'StPaulChapel', 'Philosophy', 'AlmaMater', 'Kent', 'CollegeWalk'],
                   ['LowLibrary ', 'EarlHall', 'Buell', 'CollegeWalk'], ['EarlHall', 'Lewisohn', 'Journalism&Furnald'], ['StPaulChapel ', 'Philosophy', 'Buell', ], ['Buell', 'AlmaMater'], ['Dodge', 'Lerner'],
                   ['None',], ['Journalism&Furnald', 'Carman'], ['None',], ['Lerner',]]

minimization_distance = [['Uris'], ['Uris'], ['Uris',], ['Pupin',], ['Avery'], ['Avery', ], ['OldComputerCenter', ], ['EarlHall',],
                   ['Buell'], ['Buell'], ['Chandler&Havemeyer', 'OldComputerCenter', 'EarlHall', 'Lewisohn'],
                   ['Buell',], ['Buell', ], ['LowLibrary',], ['EarlHall',], ['Buell', ], ['Avery', ], ['Buell',], ['EarlHall',], ['Buell', ], ['Buell', ], ['Dodge',],
                   ['None',], ['Journalism&Furnald',], ['None',], ['Lerner',]]


for i in buildings:
  exec('tabletake%s = [["Building Number: %s", "Values:"], ["Building Name: ", 0], ["Nearness Source:", "left"], ["Nearness Target: ", 0], \
  ["Minimization:", 0]]' % (i, i))


for i in buildings:
  exec('tabletake%s[1][1] = buildings_dict.get("%s")' %(i, i))

for i in range(26):
  exec('tabletake%s[2][1] = "%s"' %(buildings[i], nearness_source[i]))
  exec('tabletake%s[3][1] = "%s"' %(buildings[i], nearness_target[i]))
  exec('tabletake%s[4][1] = "%s"' %(buildings[i], minimization_distance[i]))

for i in buildings:
  exec('print(tabulate(tabletake%s, headers="firstrow"))' % (i))
  print('\n')



total_descriptions = [['Mid-Size', 'Wide', 'Rectangle', 'Uppermost', 'Left'], ['Mid-Size', 'Medium'], ['Largest', 'Medium', 'Asymmetrical'],
                      ['Mid-Size', 'Narrow', 'Rectangle'], ['Largest', 'Narrow'], ['Large', 'Medium', 'L', 'Upper', 'Right-most'], ['Large', 'Medium', 'L', 'Upper', 'Left-most'],
                      ['Smallest', 'Narrow'], ['Narrow', 'C'], ['Small', 'Narrow', 'Rectangle', 'Upper'], ['Small', 'Narrow', 'Rectangle', 'Mid-height'], ['Large', 'Medium', 'Square'],
                      ['Small', 'Wide'], ['Small', 'Medium'], ['Mid-Size', 'Narrow'], ['Small', 'Narrow', 'I'], ['Smallest', 'Medium', 'I'], ['Smallest', 'Medium', 'Square'],
                      ['Mid-size', 'Wide', 'C', 'Lower', 'Left'], ['Mid-size', 'Wide', 'C', 'Lower', 'Right-most'], ['Largest', 'Wide'], ['Large', 'Medium', 'Asymmetrical'],
                      ['Narrow', 'Asymmetrical'], ['Large', 'Wide'], ['Largest', 'Medium', 'I'], ['Mid-Size', 'Wide', 'Rectangle', 'Lowermost', 'Left']]


nl_descriptions = [['Wide Mid-Size Rectangle building in Uppermost Left'], ['Mid-Size Medium building'], ['Largest Medium Asymmetrical building'],
                      ['Narrow Mid-Size Rectangle building'], ['Largest Narrow building'], ['Large Medium L-shaped building in Upper Right-most'], ['Large Medium L-shaped building in Upper Left-most'],
                      ['Smallest Narrow building'], ['Narrow C-shaped building'], ['Small Narrow Rectangle building in Upper'], ['Small Narrow Rectangle building at Mid-height'], ['Large Medium Square'],
                      ['Small Wide building'], ['Small Medium building'], ['Mid-Size Narrow building'], ['Small Narrow I-shaped building'], ['Smallest Medium I-shaped building'], ['Smallest Medium Square building'],
                      ['Mid-size Wide C-shaped building in Lower Left'], ['Mid-size Wide C-shaped building in LowerRight-most'], ['Largest Wide building'], ['Large Medium Asymmetrical building'],
                      ['Narrow Asymmetrical building'], ['Large Wide building'], ['Largest Medium I-shaped building'], ['Mid-Size Wide Rectangle building in Lowermost Left']]




for i in buildings:
  exec('tabletake%s = [["Building Number: %s", "Values:"], ["Building Name: ", 0], ["Minimized:", "left"]]' % (i, i))

for i in buildings:
  exec('tabletake%s[1][1] = buildings_dict.get("%s")' %(i, i))

for i in range(26):
  exec('tabletake%s[2][1] = "%s"' %(buildings[i], nl_descriptions[i]))

for i in buildings:
  exec('print(tabulate(tabletake%s, headers="firstrow"))' % (i))
  print('\n')
