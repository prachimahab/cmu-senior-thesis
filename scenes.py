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

# scenes
surfing_indexes = [4484,31,1236,2218,5230,3028,534,176,5036,1016,2891,4053,4340,2812]
motorcycling_indexes = [1235,58,875,2459,2828,4787,5045,4034,346,1408,4064,985,2823,5160]
beach_indexes = [2947,2115,4042,386,2160,1459,4898,3219,2914,518,773,4730,2747,2492]
grocerystore_indexes = [2310,3249,3266,3125,4951,594,503,1664,2023,3503,5241,5206,4953,5081]
bus_indexes = [1760,2407,2496,3157,1277,2333,1824,4171,1326,2135,4649,4184,1409,4017]
banquethall_indexes = [4601,1553,3217,3995,5169,1924,3128,2172]
parkinggarage_indexes = [4208,520,2347,3296,1552,388,4628]
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
	total_mean = np.mean(total)
	temp = len(corr) - 1
	num_entry = (temp * (temp + 1)) / 2
	# n_total = total_mean * num_entry
	n_total = np.sum(total)

	rep = np.triu(corr[0:4, 0:4], 1)
	rep_mean = np.mean(rep)
	n_rep = rep_mean * 6
	# where n is 3
	# n_rep = np.sum(rep)

	nonrep = n_total - n_rep
	nonrep_mean = nonrep / (num_entry - 6)


	plt.imshow(corr)
	plt.colorbar()
	plt.title(name)
	plt.savefig(ROI + " " + name + "_corr.png")
	#print(total_mean, nonrep_mean, rep_mean)
	return total_mean, nonrep_mean, rep_mean


def braindat(CS, ROI, hemisphere, category_indexes):
	pull_data = grab_brain_data(CS, ROI, hemisphere, category_indexes, True)

	Surfing = category_images(pull_data, surfing_indexes)
	Motorcyling = category_images(pull_data, motorcycling_indexes)
	Beach = category_images(pull_data, beach_indexes)
	GroceryStore = category_images(pull_data, grocerystore_indexes)
	Bus = category_images(pull_data, bus_indexes)
	BanquetHall = category_images(pull_data, banquethall_indexes)
	ParkingGarage = category_images(pull_data, parkinggarage_indexes)
	Mixed = category_images(pull_data, mixed_indexes)


	surfing_corr = category_corr(Surfing, "Surfing", ROI)
	motorcycling_corr = category_corr(Motorcyling, "Motorcycling", ROI)
	beach_corr = category_corr(Beach, "Beach", ROI)
	grocerystore_corr = category_corr(GroceryStore, "Grocery Store", ROI)
	bus_corr = category_corr(Bus, "Bus", ROI)
	banquethall_corr = category_corr(BanquetHall, "Banquet Hall", ROI)
	parkinggarage_corr = category_corr(ParkingGarage, "Parking Garage", ROI)
	mixed_corr = category_corr(Mixed, "Mixed", ROI)

	# print((bear_corr, ROI))
	# plt.bar(["RepMean", "TotalMean", "NonrepMean"], 300, 0.8, birds_corr)
	# plt.show()
	return [surfing_corr, motorcycling_corr, beach_corr, grocerystore_corr,
			bus_corr, banquethall_corr, parkinggarage_corr, mixed_corr]


#braindat(CS, "EarlyVis", 'RH', mixed_indexes)
PPA_corr_array = braindat(CS, "PPA", 'RH', beach_indexes)
print(PPA_corr_array)
braindat(CS, "PPA", 'RH', beach_indexes)


def makegraphs(c_arr, name):
	TotalMeans = []
	NoRepMeans = []
	RepMeans = []
	for arr in c_arr:
		TotalMeans.append(arr[0])
		NoRepMeans.append(arr[1])
		RepMeans.append(arr[2])
	#average across all categories
	TotalMeans.append(0.022323051)
	NoRepMeans.append(0.053213249)
	RepMeans.append(0.008967971)

	plt.figure(figsize=(30, 3))  # width:30, height:3

	fig, ax = plt.subplots()
	n_groups = 9
	index = np.arange(n_groups)
	bar_width = 0.3
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
	ax.set_xticklabels(('surfing', 'motorcycling','beach', 'grocery store', 'bus',
						'banquet hall', 'parking garage', 'mixed', 'all categories'),
					   fontsize=12, rotation="vertical")
	ax.legend(['Totals', 'Noreps', 'Reps'])

	fig.tight_layout()
	print("print Here")
	plt.savefig(name + "_category_corr.png")

makegraphs(PPA_corr_array, "PPA")

#IMAGE correlations
IMAGE_Surfing = 0.11805219086220206
IMAGE_Motorcycling = 0.08820428169278242
IMAGE_Beach = 0.08464822361415354
IMAGE_GroceryStore = 0.10942736776099252
IMAGE_Bus = 0.10508092453356604
IMAGE_BanquetHall = 0.11397663996921924
IMAGE_ParkingGarage = 0.17565577916849973
IMAGE_Mixed = 0.10061605005209694

images_corrs = [0.11805219086220206,0.08820428169278242,0.08464822361415354, 0.10942736776099252,0.10508092453356604,
				0.11397663996921924, 0.17565577916849973, 0.10061605005209694 ]

def image_brain_graph(c_arr, images_corrs ):
	IMAGE_corr = np.array(images_corrs)
	Nonrep_Brain_Means = []
	Total_Image_Means = []
	for arr in c_arr:
		Nonrep_Brain_Means.append(arr[1])
	for val in IMAGE_corr:
		Total_Image_Means.append(val)

	plt.figure(figsize=(30, 3))  # width:30, height:3

	fig, ax = plt.subplots()
	n_groups = 8
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	Total_Brain_Means = ax.bar(index, Nonrep_Brain_Means, bar_width,bottom=None,align = 'center',
				   alpha=opacity, color='m',
				   label='Nonrep_Brain_Means Corr')
	Total_Image_Means = ax.bar([p + bar_width for p in index], Total_Image_Means,bar_width, align = 'edge',
					alpha=opacity, color='c',
					label='Total_Image_Means Corr')

	ax.set_xlabel('Categories')
	ax.set_ylabel('Means')
	ax.set_xticks(index)
	ax.set_xticklabels(('surfing', 'motorcycling','beach', 'grocery store', 'bus', 'banquet hall',
						'parking garage', 'mixed'), fontsize=12, rotation="vertical")
	ax.legend(['Nonrep_Brain_Means Corr', 'Total_Image_Means Corr'])

	fig.tight_layout()
	plt.savefig("Scenes-Image vs. Brain Mean Corr.png")


image_brain_graph(PPA_corr_array, images_corrs)