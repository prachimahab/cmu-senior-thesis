import numpy as np
import pickle
import json
import re
import copy
import scipy
from scipy.io import loadmat
import csv
import matplotlib.pyplot as plt



with open('COCO_img2cats.json') as f:
  	COCO_img2cats = json.load(f) # categories info


with open('imgnet_imgsyn_dict.pkl', 'rb') as f:
	stim_imgnet = pickle.load(f)


def zero_strip(s):
	if s[0] == '0':
		s = s[1:]
		return zero_strip(s)
	else:
		return s

def extract_dataset_index(stim_list, dataset='all', rep=True, return_filename=False):
	dataset_labels = copy.copy(stim_list.copy())
	COCO_idx, imagenet_idx, SUN_idx = list(), list(), list()
	COCO_cat_list, imagenet_cat_list, SUN_cat_list = list(), list(), list()
	for i, n in enumerate(stim_list):
		if 'COCO_' in n:
			if 'rep_' in n and rep is False:
				continue
			dataset_labels[i] = 'COCO'
			COCO_idx.append(i)
			n.split() #takeout \n
			COCO_id = zero_strip(str(n[21:-5]))
			COCO_cat_list.append(COCO_img2cats[COCO_id])
		elif 'JPEG' in n:
			dataset_labels[i] = 'imagenet'
			if "rep_" in n:
				if rep == False:
					continue
				n = n[4:]
			syn = stim_imgnet[n[:-1]]
			imagenet_idx.append(i)
			imagenet_cat_list.append([syn])
		else:
			dataset_labels[i] = 'SUN'
			name = n.split(".")[0]
			if "rep_" in name:
				if rep == False:
					continue
				else:
					name = name[4:]
			SUN_idx.append(i)
			if return_filename:
				SUN_cat_list.append(name)
			else:
				SUN_cat_list.append(re.split('[0-9]',name)[0])
	if rep:
		assert len(stim_list) == len(COCO_idx) + len(imagenet_idx) + len(SUN_idx)

	if dataset == 'COCO':
		return COCO_idx, COCO_cat_list
	elif dataset == 'imagenet':
		return imagenet_idx, imagenet_cat_list
	elif dataset == 'SUN':
		return SUN_idx, SUN_cat_list
	else:
		return dataset_labels


def grab_brain_data(datamat, rois, side, category_indexes, zscore=False):
	"""
	grab brain data matrix based on ROIS and sides of the brain
	"""
	#TRUE gives you the mean normalized data
	name = side+rois
	X = datamat[name]
	if zscore:
		XX = X - np.mean(X, axis=1, keepdims=True)
		XXX = XX/np.std(XX, axis=1, keepdims=True)
		return XXX
	else:
		return X

#######################################################################################################################


CS = loadmat('BOLD5000/CSI' + str(1) + '_ROIs_TR3.mat')
# CSI1_TR3_array = grab_brain_data(CS,'LOC', 'RH', True)


# Indexes for each category - first four are the repeated images

# objects
bird_indexes =[3583,178,2905,3421,710,4538,4816,5234,4255,374,5220,2404,3245,60]
train_indexes = [4072,4409,420,2219,4222,536,2013,2650,858,2165,1703,2932,4439,2332]
dog_indexes = [3528,4677,763,1410,1820,3424,4008,4746,855,525,3578,2790,3593,4667]
oranges_indexes = [726,63,1912,4355,2443,640,3509,3218,4115,87,395,1330,3261,2609]
table_indexes = [754,1225,1769,112,2493,4926,1365,1962,2052,3145,3299,4196,2901,313]
cat_indexes = [759,1627,4217,170,466,2895,2804,3310,183,4707,3859,3270,377,1708]
bear_indexes = [2722,217,1474,2086,36,3856,2190,723,496,3370,5203,3774,653,2619]
clocktower_indexes = [1222,263,472,4656,1623,4277,209,437,3620,1718,5215,2402,1735,4475]
squirrel_indexes = [3150,5117,3165,3643,538,1950,3653]
insect_indexes = [1869,3236,4264,4459,4065,2308,3253,3677,3577,236,1974,2593,2241,568]
donut_indexes = [3923,5095,498,3327,2528,2922,2671,2953,669,370,550,124]

# scenes
surfing_indexes = [4484,31,1236,2218,5230,3028,534,176,5036,1016,2891,4053,4340,2812]
motorcycling_indexes = [1235,58,875,2459,2828,4787,5045,4034,346,1408,4064,985,2823,5160]
beach_indexes = [2947,2115,4042,386,2160,1459,4898,3219,2914,518,773,4730,2747,2492]
grocerystore_indexes = [2310,3249,3266,3125,4951,594,503,1664,2023,3503,5241,5206,4953,5081]
bus_indexes = [1760,2407,2496,3157,1277,2333,1824,4171,1326,2135,4649,4184,1409,4017]
banquethall_indexes = [4601,1553,3217,3995,5169,1924,3128,2172]
parkinggarage_indexes = [4208,520,2347,3296,1552,388,4628]


# mixed categories
mixed_indexes = [4816,536,4008,5230,4787,2443,4926,2895,3856,209,1459]



def category_images(CSI1_TR3_array, category_indexes): #change from list to array using numpy
	data = []
	for val in category_indexes:
		voxel_data = CSI1_TR3_array[val]
		data.append(voxel_data)
	return np.array(data)


def category_corr(category, name, ROI):
	corr = np.corrcoef(category)
	total = np.triu(corr, 1)
	total_mean= np.mean(total)
	temp = len(corr) -1
	num_entry = (temp*(temp+1))/2
	# n_total = total_mean * num_entry
	n_total = np.sum(total)

	rep = np.triu(corr[0:4, 0:4], 1)
	rep_mean = np.mean(rep)
	n_rep = rep_mean * 6
	# where n is 3
	#n_rep = np.sum(rep)

	nonrep = n_total-n_rep
	nonrep_mean = nonrep/(num_entry- 6)
	#print(nonrep_mean)
	#nonrep_mean = nonrep/85

	plt.imshow(corr)
	#plt.colorbar()
	plt.title(name)
	plt.savefig(ROI + " " + name + "_corr.png")
	#print(total_mean, nonrep_mean, rep_mean)
	return total_mean, nonrep_mean, rep_mean


def braindat(CS, ROI, hemisphere, category_indexes):
	pull_data = grab_brain_data(CS, ROI, hemisphere, category_indexes, True)

	Bird = category_images(pull_data, bird_indexes)
	Train = category_images(pull_data, train_indexes)
	Dog = category_images(pull_data, dog_indexes)
	Surfing = category_images(pull_data, surfing_indexes)
	Motorcyling = category_images(pull_data, motorcycling_indexes)
	Oranges = category_images(pull_data, oranges_indexes)
	Table = category_images(pull_data, table_indexes)
	Cat = category_images(pull_data, cat_indexes)
	Bear = category_images(pull_data, bear_indexes)
	Clocktower = category_images(pull_data, clocktower_indexes)
	Beach = category_images(pull_data, beach_indexes)
	GroceryStore = category_images(pull_data, grocerystore_indexes)
	Bus = category_images(pull_data, bus_indexes)
	Squirrel = category_images(pull_data, squirrel_indexes)
	Insect = category_images(pull_data, insect_indexes)
	Donut = category_images(pull_data, donut_indexes)
	BanquetHall = category_images(pull_data, banquethall_indexes)
	ParkingGarage = category_images(pull_data, parkinggarage_indexes)
	Mixed = category_images(pull_data, mixed_indexes)

	birds_corr = category_corr(Bird, "Bird", ROI)
	train_corr = category_corr(Train, "Train", ROI)
	dog_corr = category_corr(Dog, "Dog", ROI)
	surfing_corr = category_corr(Surfing, "Surfing", ROI)
	motorcycling_corr = category_corr(Motorcyling, "Motorcycling", ROI)
	oranges_corr = category_corr(Oranges, "Oranges", ROI)
	table_corr = category_corr(Table, "Table", ROI)
	cat_corr = category_corr(Cat, "Cat", ROI)
	bear_corr = category_corr(Bear, "Bear", ROI)
	clocktower_corr = category_corr(Clocktower, "Clocktower", ROI)
	beach_corr = category_corr(Beach, "Beach", ROI)
	grocerystore_corr = category_corr(GroceryStore, "Grocery Store", ROI)
	bus_corr = category_corr(Bus, "Bus", ROI)
	squirrel_corr = category_corr(Squirrel, "Squirrel", ROI)
	insect_corr = category_corr(Insect, "Insect", ROI)
	donut_corr = category_corr(Donut, "Donut", ROI)
	banquethall_corr = category_corr(BanquetHall, "Banquet Hall", ROI)
	parkinggarage_corr = category_corr(ParkingGarage, "Parking Garage", ROI)
	mixed_corr = category_corr(Mixed, "Mixed", ROI)
	# print((bear_corr, ROI))
	# plt.bar(["RepMean", "TotalMean", "NonrepMean"], 300, 0.8, birds_corr)
	# plt.show()
	return [birds_corr, train_corr, dog_corr, surfing_corr, motorcycling_corr, oranges_corr, table_corr,
			cat_corr, bear_corr, clocktower_corr, beach_corr, grocerystore_corr, bus_corr, squirrel_corr, insect_corr,
			donut_corr, banquethall_corr, parkinggarage_corr, mixed_corr]


braindat(CS, "EarlyVis", 'RH', mixed_indexes)
LOC_corr_array = braindat(CS, "LOC", 'RH', mixed_indexes)
#print(LOC_corr_array)
braindat(CS, "PPA", 'RH', mixed_indexes)
birds_corr = braindat(CS, "EarlyVis", 'RH', mixed_indexes)[0]



def makegraphs(c_arr, name):
	TotalMeans = []
	NoRepMeans = []
	RepMeans = []
	for arr in c_arr:
		TotalMeans.append(arr[0])
		NoRepMeans.append(arr[1])
		RepMeans.append(arr[2])
	#average across all categories
	TotalMeans.append(0.026226194)
	NoRepMeans.append(0.062816735)
	RepMeans.append(0.021389107)

	plt.figure(figsize=(3, 4))  # width:30, height:3

	fig, ax = plt.subplots()
	n_groups = 20
	index = np.arange(n_groups)
	bar_width = 0.25
	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	total = ax.bar(index, TotalMeans, bar_width,
                alpha=opacity, color='b',
                label='TotalMeans')
	noreps = ax.bar([p + bar_width for p in index], NoRepMeans, bar_width,
                alpha=opacity, color='r',
                label='NoReps')
	reps = ax.bar([p + bar_width*2 for p in index], RepMeans, bar_width,
                alpha=opacity, color='g',
                label='Reps')


	ax.set_xlabel('Categories')
	ax.set_ylabel('Means')
	ax.set_xticks(index)
	ax.set_xticklabels(('birds', 'train', 'dog', 'surfing', 'motorcycling', 'oranges', 'table',
						'cat', 'bear', 'clocktower', 'beach','grocery store', 'bus', 'squirrel',
						'insect', 'donut', 'banquet hall', 'parking garage', 'mixed', 'all categories'),
					   fontsize=8, rotation="vertical")
	ax.legend(['Totals', 'Noreps', 'Reps'])

	fig.tight_layout()
	print("print Here")
	plt.savefig(name + "_category_corr.png")

makegraphs(LOC_corr_array, "LOC")

def diff_corr(c_arr, ROI):
	TotalMeans = []
	NonRepMeans = []
	RepMeans = []
	for arr in c_arr:
		TotalMeans.append(arr[0])
		NonRepMeans.append(arr[1])
		RepMeans.append(arr[2])
	Total_Nonrep_Diff = []
	Total_Rep_Diff = []
	Nonrep_Rep_Diff = []

	#numpy array differences

	for i in range(len(TotalMeans)):
		total_nonrep_diff = TotalMeans[i] - NonRepMeans[i]
		Total_Nonrep_Diff.append(total_nonrep_diff)
		#print(total_nonrep_diff)
	Total_Nonrep_Corr = np.corrcoef(Total_Nonrep_Diff)
	stdev1 = np.std(Total_Nonrep_Diff)
	for i in range(len(TotalMeans)):
		total_rep_diff = TotalMeans[i] - RepMeans[i]
		Total_Rep_Diff.append(total_rep_diff)
		#print(total_rep_diff)
	Total_Rep_Corr = np.corrcoef(Total_Rep_Diff)
	stdev2 = np.std(Total_Rep_Diff)
	for i in range(len(NonRepMeans)):
		nonrep_rep_diff = NonRepMeans[i] - RepMeans[i]
		Nonrep_Rep_Diff.append(nonrep_rep_diff)
		#print(nonrep_rep_diff)
	#standard deviation of the difference
	#plot the differences and then add an error bar
	#t test could be done
	print(Nonrep_Rep_Diff)
	Nonrep_Rep_Corr = np.corrcoef(Nonrep_Rep_Diff)
	stdev3 = np.std(Nonrep_Rep_Diff)


	print(stdev1, stdev2, stdev3)
	return stdev1, stdev2, stdev3

diff_corr = diff_corr(LOC_corr_array, "LOC")
print(diff_corr)

#######################################################################################################################

#IMAGE correlations
IMAGE_Bird = 0.07558363335730557
IMAGE_Train = 0.1192231944596046
IMAGE_Dog = 0.06252421166463547
IMAGE_Surfing = 0.11805219086220206
IMAGE_Motorcycling = 0.08820428169278242
IMAGE_Oranges = 0.0850816245256896
IMAGE_Table = 0.13220789183847542
IMAGE_Cat = 0.11028934415193661
IMAGE_Bear = 0.09288595455454225
IMAGE_Clocktower = 0.10389176247437619
IMAGE_Beach = 0.08464822361415354
IMAGE_GroceryStore = 0.10942736776099252
IMAGE_Bus =  0.10508092453356604
IMAGE_Squirrel = 0.07194685888679081
IMAGE_Insect = 0.06868132082245408
IMAGE_Donut = 0.1105543449947059
IMAGE_Banquet_Hall = 0.11397663996921924
IMAGE_Parking_Garage = 0.17565577916849973

IMAGE_Mixed = 0.10061605005209694

images_corrs = [0.07558363335730557,0.1192231944596046,0.06252421166463547,0.11805219086220206,0.08820428169278242,
				0.0850816245256896,0.13220789183847542,0.11028934415193661,0.09288595455454225,0.10389176247437619,
				0.08464822361415354,0.10942736776099252,0.10508092453356604,0.07194685888679081,0.06868132082245408,
				0.1105543449947059,0.11397663996921924,0.17565577916849973, 0.10061605005209694]

def image_brain_graph(c_arr, images_corrs ):
	IMAGE_corr = np.array(images_corrs)
	Nonrep_Brain_Means = []
	Total_Image_Means = []
	for arr in c_arr:
		Nonrep_Brain_Means.append(arr[1])
	for val in IMAGE_corr:
		Total_Image_Means.append(val)
	r = np.corrcoef(Nonrep_Brain_Means, Total_Image_Means)
	print(r, "r")
	plt.figure(figsize=(100, 20))  # width:30, height:3

	fig, ax = plt.subplots()
	n_groups = 19
	index = np.arange(n_groups)
	bar_width = 0.3
	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	Total_Brain_Means = ax.bar(index, Nonrep_Brain_Means, bar_width,bottom=None,align = 'center',
				   alpha=opacity, color='m',
				   label='Nonrep Brain Means')
	Total_Image_Means = ax.bar([p + bar_width for p in index], Total_Image_Means,bar_width, align = 'edge',
					alpha=opacity, color='c',
					label='Total Image Means')
	#yerror = [standard deviations]


	ax.set_xlabel('Categories')
	ax.set_ylabel('Means')
	ax.set_xticks(index)
	ax.set_xticklabels(('birds', 'train', 'dog', 'surfing', 'motorcycling', 'oranges', 'table',
						'cat', 'bear', 'clocktower', 'beach', 'grocery store','bus', 'squirrel',
						'insect', 'donut', 'banquet hall', 'parking garage','mixed'),
					   fontsize=12, rotation=45)
	# ax.set_xticklabels(('birds', 'train', 'dog', 'surfing', 'motorcycling', 'oranges', 'table',
	# 					'cat', 'bear', 'clocktower', 'beach', 'grocery store', 'bus', 'squirrel',
	# 					'insect', 'donut', 'banquet hall', 'parking garage', 'mixed'),
	# 				   fontsize=12, rotation="vertical")
	ax.legend(['Nonrep Brain Means', 'Total Image Means'])

	fig.tight_layout()
	plt.savefig("Image vs. Brain Mean Corr.png")


image_brain_graph(LOC_corr_array, images_corrs)

# def brainimagecorr(c_arr, images_corrs):
# 	IMAGE_corr = np.array(images_corrs)
# 	Nonrep_Brain_Means = []
# 	Total_Image_Means = []
# 	for arr in c_arr:
# 		Nonrep_Brain_Means.append(arr[1])
# 	for val in IMAGE_corr:
# 		Total_Image_Means.append(val)
# 	r = np.corrcoef(Total_Image_Means, Nonrep_Brain_Means)
# 	return r
#
# r = brainimagecorr(LOC_corr_array, images_corrs)
# print(r, "r")

